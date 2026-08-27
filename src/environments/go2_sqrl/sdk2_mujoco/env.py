"""RL-X vector-environment adapter backed only by Unitree SDK2 topics."""

from __future__ import annotations

import logging
import time

import numpy as np
from gymnasium.spaces import Box

from rl_x.environments.safety_rollout import InvalidTransitionError

from ..common.action import ActionMapper, project_actions_from_observation
from ..common.estimation.velocity import (
    VelocityEstimator,
    quaternion_rotation_matrix_wxyz,
    velocity_estimator_config_from,
)
from ..common.observation import ObservationBuilder
from ..common.reward import compute_reward
from ..common.reward import BASE_HEIGHT_TARGET
from ..common.specs import (
    ACTION_SIZE,
    CONTROL_DT,
    DEFAULT_JOINT_POSITION,
    OBSERVATION_SIZE,
    PHYSICS_DT,
    PHYSICS_STEPS_PER_ACTION,
)
from ..common.termination import EpisodeTracker
from ..common.manifest import (
    build_manifest,
    validate_manifest,
    validate_transfer_manifest as validate_transfer_manifest_contract,
)
from .fall_detector import FallDetector
from .general_properties import GeneralProperties
from .reset_controller import MujocoResetController
from .sdk_client import SDKClient
from .state_buffer import StateBufferError, StateTimeout


logger = logging.getLogger("rl_x")


class Go2SDKMujocoEnv:
    general_properties = GeneralProperties
    policy_observation_indices = np.arange(OBSERVATION_SIZE)
    critic_observation_indices = np.arange(OBSERVATION_SIZE)
    safety_critic_observation_indices = np.arange(OBSERVATION_SIZE)

    def __init__(
        self,
        config,
        client: SDKClient | None = None,
        role: str = "train",
        reset_controller: MujocoResetController | None = None,
    ):
        environment = config.environment
        if int(environment.policy_frames) != PHYSICS_STEPS_PER_ACTION:
            raise ValueError(
                "MuJoCo control windows must contain exactly "
                f"{PHYSICS_STEPS_PER_ACTION} LowState frames"
            )
        if not np.isclose(
            float(environment.policy_period_seconds), CONTROL_DT,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"MuJoCo policy_period_seconds must remain {CONTROL_DT}s"
            )
        self.config = environment
        self.role = role
        runner = getattr(config, "runner", None)
        runner_mode = str(getattr(runner, "mode", "train"))
        self._startup_reset_role = "eval" if runner_mode == "test" else "train"
        self.nr_envs = 1
        self.num_envs = 1
        self.single_observation_space = Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.single_action_space = Box(
            low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.client = client or SDKClient(environment.domain_id, environment.interface)
        if reset_controller is None and isinstance(self.client, SDKClient):
            reset_controller = MujocoResetController(
                window_title=environment.mujoco_window_title
            )
        self.reset_controller = reset_controller
        self.action_mapper = ActionMapper()
        self.observation_builder = ObservationBuilder(
            VelocityEstimator(
                dt=PHYSICS_DT,
                config=velocity_estimator_config_from(environment),
            )
        )
        self.fall_detector = FallDetector(
            environment.fall_angle_threshold,
            environment.fall_consecutive_frames,
            environment.reset_min_base_height,
        )
        self.episode = EpisodeTracker(environment.episode_steps)
        self._last_tick: int | None = None
        self._generation = 0
        self._last_observation: np.ndarray | None = None
        self._policy_blend_elapsed = 0.0
        self._initial_simulator_reset_done = False
        self._previous_reward_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._policy_window_callback = None

    @staticmethod
    def project_actions(states, actions):
        """Project actions using only the previous target encoded in ``states``.

        This hook is side-effect free so an algorithm may apply it to replay,
        policy candidates, and Q queries without advancing the DDS environment.
        NumPy arrays are returned as NumPy; Array-API/JAX inputs retain their
        array namespace when the installed array implementation exposes it.
        """

        return project_actions_from_observation(states, actions)

    def set_policy_window_callback(self, callback):
        """Run one learner update while the external simulator executes an action."""

        if callback is not None and not callable(callback):
            raise TypeError("policy window callback must be callable")
        if self._policy_window_callback is not None:
            raise RuntimeError("a policy window callback is already pending")
        self._policy_window_callback = callback

    def _wait_window(self, count: int | None = None):
        count = int(count or self.config.policy_frames)
        try:
            frames = self.client.state_buffer.wait_for_frames(
                count=count,
                after_tick=self._last_tick,
                timeout=float(self.config.state_timeout),
                generation=self._generation,
            )
        except StateBufferError as exc:
            raise InvalidTransitionError(str(exc)) from exc
        self._last_tick = int(frames[-1].tick)
        return frames

    def _reset_pose_ready(self, state) -> bool:
        joint_error = np.max(
            np.abs(
                np.asarray(state.joint_q, dtype=np.float32)
                - DEFAULT_JOINT_POSITION
            )
        )
        max_joint_velocity = np.max(
            np.abs(np.asarray(state.joint_dq, dtype=np.float32))
        )
        orientation_ready = self.fall_detector.is_stable(state.imu_quat)
        joints_ready = joint_error <= float(self.config.reset_joint_tolerance)
        joints_still = max_joint_velocity <= float(
            self.config.reset_max_joint_velocity
        )

        # Roll and pitch alone cannot distinguish a standing robot from one
        # lying flat on its belly.  The simulator-only high-state topic exposes
        # base height without leaking it into the policy observation.
        height_ready = True
        latest_training_state = getattr(self.client, "latest_training_state", None)
        truth = latest_training_state() if callable(latest_training_state) else None
        if truth is not None and truth.base_position is not None:
            position = np.asarray(truth.base_position, dtype=np.float32)
            height_ready = (
                position.size >= 3
                and np.isfinite(position[2])
                and float(position[2]) >= float(self.config.reset_min_base_height)
            )
        return orientation_ready and joints_ready and joints_still and height_ready

    def _duration_steps(self, seconds: float) -> int:
        period = float(self.config.policy_period_seconds)
        if period <= 0.0:
            raise ValueError("environment.policy_period_seconds must be positive")
        return max(0, int(np.ceil(float(seconds) / period)))

    def _publish_and_wait(self, target: np.ndarray):
        target = np.asarray(target, dtype=np.float32)
        # DDS may briefly deliver queued pre-reset ticks after the first new
        # low tick. StateBuffer correctly records another generation change,
        # but reset interpolation should resynchronize rather than treating it
        # as a policy transition. Runtime ``step`` remains strict.
        # CycloneDDS can interleave several queued pre-reset samples with the
        # new epoch.  A fixed retry count is therefore racy: every old/new
        # boundary legitimately advances StateBuffer's generation.  Bound the
        # recovery by wall time instead and require one complete policy window
        # from a stable generation before continuing the stand-up sequence.
        deadline = time.monotonic() + max(
            float(self.config.state_timeout),
            10.0 * float(self.config.policy_period_seconds),
        )
        resynchronizations = 0
        while True:
            generation = self.client.state_buffer.generation
            self._generation = generation
            self._last_tick = self.client.state_buffer.last_tick
            if isinstance(self.client, SDKClient):
                self.client.publish_joint_target(
                    target,
                    kp=float(self.config.reset_kp),
                    kd=float(self.config.reset_kd),
                )
            else:
                self.client.publish_joint_target(target)
            try:
                return self._wait_window()[-1]
            except InvalidTransitionError:
                current_generation = self.client.state_buffer.generation
                if (
                    current_generation == generation
                    or time.monotonic() >= deadline
                ):
                    raise
                resynchronizations += 1
                if resynchronizations == 1:
                    logger.info(
                        "MuJoCo tick generation advanced during reset pose; "
                        "draining queued pre-reset LowState samples."
                    )
                self.client.state_buffer.clear_error()
                self._last_tick = None
                self.fall_detector.reset()

    def _wait_for_simulator_restart(self, reason: str) -> None:
        logger.warning("%s Press Backspace in the unitree_mujoco window to reset.", reason)
        timeout = float(self.config.manual_reset_timeout)
        timeout = None if timeout < 0 else timeout
        self._generation = self.client.state_buffer.wait_for_restart(
            self._generation, timeout=timeout
        )
        settle_seconds = float(self.config.reset_post_restart_settle_seconds)
        if settle_seconds > 0.0:
            time.sleep(settle_seconds)
        self._generation = self.client.state_buffer.generation
        self.client.state_buffer.clear_error()
        self._last_tick = None
        self.fall_detector.reset()

    def _auto_reset_simulator(self, reason: str, delay_seconds: float = 0.0) -> None:
        if self.reset_controller is None:
            self._wait_for_simulator_restart(reason)
            return
        if delay_seconds > 0.0:
            logger.warning(
                f"{reason} Automatic MuJoCo reset in {delay_seconds:.1f} seconds."
            )
            time.sleep(delay_seconds)
        else:
            logger.info(f"{reason} Automatically resetting MuJoCo.")

        generation = self.client.state_buffer.generation
        self.reset_controller.reset()
        try:
            self._generation = self.client.state_buffer.wait_for_restart(
                generation,
                timeout=float(self.config.auto_reset_timeout_seconds),
            )
        except StateTimeout as exc:
            raise RuntimeError(
                "MuJoCo received the automatic reset key, but no simulator "
                "tick restart was observed."
            ) from exc
        settle_seconds = float(self.config.reset_post_restart_settle_seconds)
        if settle_seconds > 0.0:
            time.sleep(settle_seconds)
        self._generation = self.client.state_buffer.generation
        self.client.state_buffer.clear_error()
        self._last_tick = None
        self.fall_detector.reset()

    def _interpolate_reset_pose(self, state, start, target, seconds):
        steps = self._duration_steps(seconds)
        start = np.asarray(start, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        for index in range(steps):
            alpha = float(index + 1) / float(steps)
            state = self._publish_and_wait((1.0 - alpha) * start + alpha * target)
        return state

    def _hold_reset_pose(self, initial_state):
        """Linearly stand up before handing control directly to the policy."""

        state = initial_state
        while True:
            pose_1 = np.asarray(self.config.standup_pose_1, dtype=np.float32)
            state = self._interpolate_reset_pose(
                state,
                state.joint_q,
                pose_1,
                self.config.standup_phase_1_seconds,
            )
            state = self._interpolate_reset_pose(
                state,
                pose_1,
                DEFAULT_JOINT_POSITION,
                self.config.standup_phase_2_seconds,
            )
            for _ in range(self._duration_steps(self.config.standup_hold_seconds)):
                state = self._publish_and_wait(DEFAULT_JOINT_POSITION)

            for _ in range(
                self._duration_steps(self.config.reset_sync_timeout_seconds)
            ):
                if self._reset_pose_ready(state):
                    return state
                state = self._publish_and_wait(DEFAULT_JOINT_POSITION)
            if self._reset_pose_ready(state):
                return state

            self._auto_reset_simulator(
                "Go2 did not reach the standing pose after interpolation."
            )
            state = self.client.state_buffer.latest_state
            if state is None:
                state = self._wait_window(count=1)[-1]

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.client.start()
        self._generation = self.client.state_buffer.generation
        if (
            self.role == self._startup_reset_role
            and bool(self.config.auto_reset_on_start)
            and self.reset_controller is not None
            and not self._initial_simulator_reset_done
        ):
            # Observe a pre-reset tick first so StateBuffer can recognize the
            # simulator tick rollback caused by Backspace.
            self._last_tick = self.client.state_buffer.last_tick
            if self._last_tick is None:
                self._wait_window(count=1)
            self._auto_reset_simulator("Initial policy startup.")
            self._initial_simulator_reset_done = True
        self.action_mapper.reset()
        self.observation_builder.reset()
        self.fall_detector.reset()
        self.episode.reset()
        self._policy_blend_elapsed = 0.0
        self._previous_reward_action.fill(0.0)
        # Capture the measured reset pose before issuing any command. Stand-up
        # interpolation begins from this feedback, not from an assumed pose.
        state = self.client.state_buffer.latest_state
        if state is None:
            self._last_tick = None
            state = self._wait_window(count=1)[-1]
        else:
            self._last_tick = int(state.tick)
        state = self._hold_reset_pose(state)
        observation, _ = self.observation_builder.build(state)
        self._last_observation = observation
        return observation[None, :], {}

    def _manual_failure_reset(self):
        if bool(self.config.auto_reset_after_fall):
            self._auto_reset_simulator(
                "Go2 fall detected.",
                delay_seconds=float(self.config.fall_auto_reset_delay_seconds),
            )
        else:
            self._wait_for_simulator_restart("Go2 fall detected.")
        return self.reset()[0][0]

    def _logical_reset(self, observation):
        """Start a new accounting episode without breaking physical continuity.

        A time limit does not reset the external simulator.  Controller memory,
        the KF, quaternion continuity, the previous target/action, and the fall
        debounce state must therefore remain continuous as well.  Rebuilding
        the observation here would also integrate the final LowState twice.
        """

        self.episode.reset()
        return np.asarray(observation, dtype=np.float32).copy()

    def _time_limit_reset(self, observation):
        """Return the first observation of an independent physical rollout."""

        if bool(self.config.auto_reset_on_time_limit) and self.reset_controller is not None:
            self._auto_reset_simulator("Go2 episode time limit reached.")
            return self.reset()[0][0]
        # Dependency-injected unit tests and headless SDK deployments without a
        # reset controller retain a physically continuous fallback.  Crucially,
        # it also retains all controller/estimator memory.
        return self._logical_reset(observation)

    def step(self, actions):
        action = np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_SIZE)[0]
        blend_seconds = float(self.config.policy_blend_seconds)
        if blend_seconds <= 0.0:
            alpha = 1.0
        else:
            self._policy_blend_elapsed = min(
                blend_seconds,
                self._policy_blend_elapsed
                + float(self.config.policy_period_seconds),
            )
            alpha = self._policy_blend_elapsed / blend_seconds
        # The home action is zero in the normalized contract because zero
        # maps exactly to DEFAULT_JOINT_POSITION.
        home_action = np.zeros_like(action)
        blended_action = np.asarray(
            (1.0 - alpha) * home_action + alpha * action,
            dtype=np.float32,
        )
        mapped = self.action_mapper.apply(blended_action)
        reward_action = np.clip(mapped.raw_action, -1.0, 1.0)
        # JAX updates can take longer than one control interval while the C++
        # simulator continues publishing LowState.  Discard that backlog by
        # anchoring the next ten-frame window at the latest pre-command tick.
        self._last_tick = self.client.state_buffer.last_tick
        if isinstance(self.client, SDKClient):
            self.client.publish_joint_target(
                mapped.q_target,
                kp=float(self.config.policy_kp),
                kd=float(self.config.policy_kd),
            )
        else:
            self.client.publish_joint_target(mapped.q_target)
        self.observation_builder.set_previous_q_target(mapped.q_target)
        callback = self._policy_window_callback
        self._policy_window_callback = None
        if callback is not None:
            callback()
        frames = self._wait_window()

        failure = False
        for frame in frames:
            failure = self.fall_detector.update(frame.imu_quat) or failure
        final_state = frames[-1]
        observation, estimated_body_velocity = self.observation_builder.build_many(
            frames
        )

        truth = self.client.latest_training_state()
        if truth is None:
            rotation = quaternion_rotation_matrix_wxyz(final_state.imu_quat)
            world_velocity = rotation @ estimated_body_velocity
        else:
            world_velocity = np.asarray(truth.world_velocity)
        rotation = quaternion_rotation_matrix_wxyz(final_state.imu_quat)
        truth_body_velocity = rotation.T @ np.asarray(world_velocity)
        base_height = BASE_HEIGHT_TARGET
        if truth is not None and truth.base_position is not None:
            base_height = float(np.asarray(truth.base_position)[2])
            failure = self.fall_detector.update_base_height(base_height) or failure
        terms = compute_reward(
            world_velocity,
            final_state.imu_quat,
            final_state.imu_gyro,
            final_state.joint_q,
            reward_action,
            self._previous_reward_action,
            float(self.config.target_velocity_x),
            base_height=base_height,
        )
        self._previous_reward_action = reward_action.copy()
        terminated, truncated = self.episode.advance(terms.total, failure)
        estimator = self.observation_builder.velocity_estimator
        innovation_squared = getattr(estimator, "last_innovation_squared", None)
        support_confidence = getattr(estimator, "last_support_confidence", None)
        covariance = getattr(estimator, "covariance", None)

        info = {
            "failure": np.asarray([int(failure)], dtype=np.float32),
            "applied_action": mapped.applied_action[None, :],
            "policy_blend_alpha": np.asarray([alpha], dtype=np.float32),
            "forward_velocity": np.asarray(
                [truth_body_velocity[0]], dtype=np.float32
            ),
            "estimated_forward_velocity": np.asarray(
                [estimated_body_velocity[0]], dtype=np.float32
            ),
            "target_velocity_error": np.asarray(
                [
                    abs(
                        float(self.config.target_velocity_x)
                        - float(truth_body_velocity[0])
                    )
                ],
                dtype=np.float32,
            ),
            "velocity_estimation_error": np.asarray(
                [np.linalg.norm(truth_body_velocity - estimated_body_velocity)],
                dtype=np.float32,
            ),
            "velocity_estimator/measurement_accepted": np.asarray(
                [float(getattr(estimator, "last_measurement_accepted", False))],
                dtype=np.float32,
            ),
            "velocity_estimator/innovation_nis": np.asarray(
                [np.nan if innovation_squared is None else innovation_squared],
                dtype=np.float32,
            ),
            "velocity_estimator/support_confidence_sum": np.asarray(
                [
                    np.nan
                    if support_confidence is None
                    else np.sum(support_confidence)
                ],
                dtype=np.float32,
            ),
            "velocity_estimator/covariance_trace": np.asarray(
                [np.nan if covariance is None else np.trace(covariance)],
                dtype=np.float32,
            ),
            **{key: np.asarray([value], dtype=np.float32) for key, value in terms.as_dict().items()},
        }

        terminal_info = None
        final_observation = None
        if terminated or truncated:
            final_observation = observation.copy()
            terminal_info = {
                **{key: float(value[0]) for key, value in info.items() if np.asarray(value).size == 1},
                "episode_return": self.episode.episode_return,
                "episode_length": self.episode.steps,
            }
            if terminated:
                observation = self._manual_failure_reset()
            else:
                observation = self._time_limit_reset(observation)
            info["episode_return"] = np.asarray(
                [terminal_info["episode_return"]], dtype=np.float32
            )
            info["episode_length"] = np.asarray(
                [terminal_info["episode_length"]], dtype=np.float32
            )

        info["final_observation"] = [final_observation]
        info["final_info"] = [terminal_info]
        self._last_observation = observation
        return (
            observation[None, :],
            np.asarray([terms.total], dtype=np.float32),
            np.asarray([terminated], dtype=bool),
            np.asarray([truncated], dtype=bool),
            info,
        )

    def get_final_observation_at_index(self, info, index):
        return info["final_observation"][index]

    def get_final_info_value_at_index(self, info, key, index):
        final_info = info["final_info"][index]
        if final_info is None:
            raise KeyError(f"No final info for environment {index}")
        return final_info[key]

    def get_logging_info_dict(self, info):
        ignored = {"failure", "applied_action", "final_observation", "final_info"}
        return {
            key: np.asarray(value).reshape(-1).tolist()
            for key, value in info.items()
            if key not in ignored and not isinstance(value, list)
        }

    def close(self):
        self.client.close()

    def checkpoint_manifest(self, normalizer=None):
        return build_manifest(
            normalizer,
            fall_angle_threshold=float(self.config.fall_angle_threshold),
            fall_consecutive_frames=int(self.config.fall_consecutive_frames),
            target_velocity_x=float(self.config.target_velocity_x),
        )

    def validate_checkpoint_manifest(self, manifest, normalizer=None):
        validate_manifest(manifest, self.checkpoint_manifest(normalizer))

    def validate_transfer_manifest(self, manifest, normalizer=None):
        validate_transfer_manifest_contract(
            manifest, self.checkpoint_manifest(normalizer)
        )
