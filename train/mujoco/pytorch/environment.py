"""SDK2/MuJoCo implementation of the shared Go2 environment."""
from __future__ import annotations

import logging
import time

import numpy as np
from ml_collections import config_dict

from sqrl.sac.environment import InvalidTransitionError
from train.common.base import (
    ACTION_SIZE, ACTION_SPEC, CONTROL_DT, DEFAULT_JOINT_POSITION, Go2Environment,
    PHYSICS_DT, PHYSICS_STEPS_PER_ACTION, ActionMapper,
    configure_failure_detection, format_policy_io_contract,
)
from train.common.estimation import (
    VelocityEstimator, configure_velocity_estimator,
    foot_position_velocity_body, quaternion_rotation_matrix_wxyz,
    velocity_estimator_config_from,
)
from .sdk import MujocoResetController, SDKClient, StateBufferError, StateTimeout
from train.common.task import (
    BASE_HEIGHT_TARGET, EpisodeTracker, ObservationBuilder, compute_reward,
    local_base_clearance, quaternion_to_rpy_wxyz,
    swing_foot_clearance_error,
)

def get_config(environment_name):
    config = config_dict.ConfigDict()
    config.name = environment_name
    config.seed = 0
    config.nr_envs = 1
    config.domain_id = 1
    config.interface = "lo"
    config.policy_frames = 10
    # LowState is published at 500 Hz; one policy/learner iteration consumes
    # ten fresh physical ticks while DDS transport remains on the host.
    config.policy_period_seconds = 0.02
    configure_velocity_estimator(config)
    configure_failure_detection(config)
    # Runtime policy PD. Defaults reproduce the Isaac/checkpoint contract;
    # command-line overrides are MuJoCo-only sensitivity experiments.
    config.policy_kp = 25.0
    config.policy_kd = 0.5
    config.state_timeout = 1.0
    config.manual_reset_timeout = -1.0
    config.auto_reset_on_start = True
    config.auto_reset_after_fall = True
    config.fall_auto_reset_delay_seconds = 1.0
    config.auto_reset_timeout_seconds = 10.0
    # X11 synthetic key events can occasionally be dropped by the visible
    # simulator window. Re-arm tick rollback detection and retry the shortcut
    # instead of terminating a long online run on one missed event.
    config.auto_reset_attempts = 3
    config.mujoco_window_title = "MuJoCo"
    # Two-stage linear stand-up copied from the proven Go2 controller in the
    # reference repository: current pose -> folded/crouched keyframe -> home.
    config.standup_pose_1 = [
        0.0, 1.36, -2.65,
        0.0, 1.36, -2.65,
        -0.2, 1.36, -2.65,
        0.2, 1.36, -2.65,
    ]
    config.standup_phase_1_seconds = 1.0
    config.standup_phase_2_seconds = 1.0
    config.standup_hold_seconds = 2.0
    config.reset_sync_timeout_seconds = 3.0
    config.reset_kp = 60.0
    # Kd=5 drives the torque-clipped SDK bridge into a persistent limit cycle
    # on this 2 ms MuJoCo model.  Kd=1 settles the same stand-up trajectory
    # before policy takeover while keeping enough damping for foot impacts.
    config.reset_kd = 1.0
    # Stand-up completes before the first policy step, so no second blend is
    # applied to the policy action.
    config.policy_blend_seconds = 0.0
    config.reset_joint_tolerance = 0.20
    # Position alone can look ready while the legs are still oscillating.
    config.reset_max_joint_velocity = 0.5
    config.reset_min_base_height = 0.20
    config.episode_steps = 500
    # The policy has no command input, so evaluation must use the fixed velocity
    # objective on which it was pre-trained.
    config.target_velocity_x = 0.5

    config.kp = 25.0
    config.kd = 0.5

    return config

class FallDetector:
    def __init__(
        self,
        angle_threshold: float = 0.8,
        consecutive_frames: int = 5,
        min_base_clearance: float = 0.18,
    ):
        self.angle_threshold = float(angle_threshold)
        self.consecutive_frames = int(consecutive_frames)
        self.min_base_clearance = float(min_base_clearance)
        self._count = 0
        self._low_height_count = 0
        self.last_tilt_failure = False
        self.last_height_failure = False

    def reset(self) -> None:
        self._count = 0
        self._low_height_count = 0
        self.last_tilt_failure = False
        self.last_height_failure = False

    def update(self, quaternion) -> bool:
        roll, pitch, _ = quaternion_to_rpy_wxyz(quaternion)
        fallen = abs(roll) > self.angle_threshold or abs(pitch) > self.angle_threshold
        self._count = self._count + 1 if fallen else 0
        self.last_tilt_failure = self._count >= self.consecutive_frames
        return self.last_tilt_failure

    def update_base_clearance(self, clearance: float | None) -> bool:
        if clearance is None or not np.isfinite(clearance):
            self._low_height_count = 0
            self.last_height_failure = False
            return False
        if float(clearance) < self.min_base_clearance:
            self._low_height_count += 1
        else:
            self._low_height_count = 0
        self.last_height_failure = (
            self._low_height_count >= self.consecutive_frames
        )
        return self.last_height_failure

    def is_stable(self, quaternion) -> bool:
        roll, pitch, _ = quaternion_to_rpy_wxyz(quaternion)
        return abs(roll) < 0.25 and abs(pitch) < 0.25

logger = logging.getLogger("go2_sdk2_mujoco")


class Go2MujocoEnv(Go2Environment):
    @classmethod
    def create_pair(cls, config):
        client = SDKClient(config.environment.domain_id, config.environment.interface)
        reset = MujocoResetController(window_title=config.environment.mujoco_window_title)
        return cls(config, client, "train", reset), cls(config, client, "eval", reset)

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
            float(environment.policy_period_seconds),
            CONTROL_DT,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"MuJoCo policy_period_seconds must remain {CONTROL_DT}s"
            )
        super().__init__(config, 1)
        self.role = role
        runner = getattr(config, "runner", None)
        runner_mode = str(getattr(runner, "mode", "train"))
        self._startup_reset_role = "eval" if runner_mode == "test" else "train"
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
            environment.fall_min_base_clearance,
        )
        self.episode = EpisodeTracker(environment.episode_steps)
        self._last_tick: int | None = None
        self._generation = 0
        self._last_observation: np.ndarray | None = None
        self._policy_blend_elapsed = 0.0
        self._initial_simulator_reset_done = False
        self._previous_reward_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        logger.info(
            "\n" + format_policy_io_contract(environment.target_velocity_x)
        )

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
        self._last_tick = self.client.state_buffer.last_tick
        target = np.asarray(target, dtype=np.float32)
        if isinstance(self.client, SDKClient):
            self.client.publish_joint_target(
                target,
                kp=float(self.config.reset_kp),
                kd=float(self.config.reset_kd),
            )
        else:
            self.client.publish_joint_target(target)
        return self._wait_window()[-1]

    def _wait_for_simulator_restart(self, reason: str) -> None:
        logger.warning(
            f"{reason} Press Backspace in the unitree_mujoco window to reset."
        )
        timeout = float(self.config.manual_reset_timeout)
        timeout = None if timeout < 0 else timeout
        self.client.state_buffer.arm_restart()
        try:
            self._generation = self.client.state_buffer.wait_for_restart(
                self._generation, timeout=timeout
            )
        except Exception:
            self.client.state_buffer.cancel_restart()
            raise
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

        # Arm the SDK bridge before resetting physics.  LowCmd is retained by
        # unitree_mujoco while mjData is reset, so the very first post-reset
        # step already holds the same home pose as the MJCF keyframe.
        if isinstance(self.client, SDKClient):
            self.client.publish_joint_target(
                DEFAULT_JOINT_POSITION,
                kp=float(self.config.reset_kp),
                kd=float(self.config.reset_kd),
            )
        else:
            self.client.publish_joint_target(DEFAULT_JOINT_POSITION)

        attempts = max(1, int(self.config.auto_reset_attempts))
        last_timeout = None
        for attempt in range(1, attempts + 1):
            generation = self.client.state_buffer.generation
            self.client.state_buffer.arm_restart()
            self.reset_controller.reset()
            try:
                self._generation = self.client.state_buffer.wait_for_restart(
                    generation,
                    timeout=float(self.config.auto_reset_timeout_seconds),
                )
                break
            except StateTimeout as exc:
                last_timeout = exc
                self.client.state_buffer.cancel_restart()
                if attempt < attempts:
                    logger.warning(
                        "MuJoCo automatic reset attempt %d/%d produced no tick "
                        "restart; retrying.",
                        attempt,
                        attempts,
                    )
        else:
            raise RuntimeError(
                f"MuJoCo received {attempts} automatic reset key attempts, but "
                "no simulator tick restart was observed."
            ) from last_timeout
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
        """Hold an exact home reset, with interpolation as a recovery fallback."""

        state = initial_state
        # A patched unitree_mujoco reset loads the named home keyframe.  Keep
        # commanding it while DDS settles.  Once home is observed, retain the
        # normal hold period so gravity/foot-contact transients finish before
        # the first policy observation is built.
        home_ready = False
        for _ in range(
            max(1, self._duration_steps(self.config.reset_sync_timeout_seconds))
        ):
            if self._reset_pose_ready(state):
                home_ready = True
                break
            state = self._publish_and_wait(DEFAULT_JOINT_POSITION)
        if self._reset_pose_ready(state):
            home_ready = True
        if home_ready:
            for _ in range(self._duration_steps(self.config.standup_hold_seconds)):
                state = self._publish_and_wait(DEFAULT_JOINT_POSITION)
            if self._reset_pose_ready(state):
                return state

        # Compatibility fallback for a simulator reset that did not load the
        # home keyframe (or for recovery from a genuinely fallen state).
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
        """Reset accounting without integrating the final physics frame twice."""

        self.episode.reset()
        return np.asarray(observation, dtype=np.float32).copy()

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
        # Learner updates can take longer than one control interval while the C++
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
        frames = self._wait_window()

        failure = False
        tilt_failure = False
        for frame in frames:
            frame_tilt_failure = self.fall_detector.update(frame.imu_quat)
            tilt_failure = frame_tilt_failure or tilt_failure
            failure = frame_tilt_failure or failure
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
        reward_body_velocity = rotation.T @ np.asarray(world_velocity)
        base_clearance = BASE_HEIGHT_TARGET
        foot_clearance_error = 0.0
        foot_clearance = np.full(4, np.nan, dtype=np.float64)
        if truth is not None and truth.base_position is not None:
            # The canonical SDK2 MuJoCo scene has a zero-height ground plane,
            # so world z is exactly the local terrain clearance.
            base_clearance = float(
                local_base_clearance(np.asarray(truth.base_position)[2], 0.0)
            )
            height_failure = self.fall_detector.update_base_clearance(
                base_clearance
            )
            failure = height_failure or failure
            foot_position_body, joint_foot_velocity_body = (
                foot_position_velocity_body(
                    final_state.joint_q, final_state.joint_dq
                )
            )
            relative_foot_velocity_body = (
                joint_foot_velocity_body
                + np.cross(
                    np.asarray(final_state.imu_gyro, dtype=np.float64)[None, :],
                    foot_position_body,
                )
            )
            foot_position_world = (
                np.asarray(truth.base_position, dtype=np.float64)[None, :]
                + (rotation @ foot_position_body.T).T
            )
            foot_velocity_world = (
                np.asarray(world_velocity, dtype=np.float64)[None, :]
                + (rotation @ relative_foot_velocity_body.T).T
            )
            foot_clearance = foot_position_world[:, 2]
            foot_clearance_error = swing_foot_clearance_error(
                foot_clearance,
                np.linalg.norm(foot_velocity_world[:, :2], axis=-1),
            )
        else:
            height_failure = False
        terms = compute_reward(
            world_velocity,
            final_state.imu_quat,
            final_state.imu_gyro,
            final_state.joint_q,
            reward_action,
            self._previous_reward_action,
            float(self.config.target_velocity_x),
            base_clearance=base_clearance,
            foot_clearance_error=foot_clearance_error,
        )
        self._previous_reward_action = reward_action.copy()
        terminated, truncated = self.episode.advance(terms.total, failure)
        torque_saturation_ratio = np.nan
        if truth is not None and truth.actuator_torque is not None:
            torque_saturation_ratio = float(
                np.mean(
                    np.abs(np.asarray(truth.actuator_torque))
                    > 0.95 * ACTION_SPEC.effort_limit
                )
            )
        estimator = self.observation_builder.velocity_estimator
        innovation_squared = getattr(estimator, "last_innovation_squared", None)
        support_confidence = getattr(estimator, "last_support_confidence", None)
        covariance = getattr(estimator, "covariance", None)

        info = {
            "failure": np.asarray([int(failure)], dtype=np.float32),
            "failure/tilt": np.asarray([int(tilt_failure)], dtype=np.float32),
            "failure/height": np.asarray([int(height_failure)], dtype=np.float32),
            "applied_action": mapped.applied_action[None, :],
            "policy_blend_alpha": np.asarray([alpha], dtype=np.float32),
            "mean_foot_clearance": np.asarray(
                [np.nan if np.isnan(foot_clearance).all() else np.nanmean(foot_clearance)],
                dtype=np.float32,
            ),
            "max_foot_clearance": np.asarray(
                [np.nan if np.isnan(foot_clearance).all() else np.nanmax(foot_clearance)],
                dtype=np.float32,
            ),
            "action_saturation_ratio": np.asarray(
                [np.mean(np.abs(reward_action) > 0.98)], dtype=np.float32
            ),
            "torque_saturation_ratio": np.asarray(
                [torque_saturation_ratio], dtype=np.float32
            ),
            "estimated_forward_velocity": np.asarray(
                [estimated_body_velocity[0]], dtype=np.float32
            ),
            "reward_uses_simulator_truth": np.asarray(
                [float(truth is not None)],
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
        if truth is not None:
            info.update(
                {
                    "forward_velocity": np.asarray(
                        [reward_body_velocity[0]], dtype=np.float32
                    ),
                    "target_velocity_error": np.asarray(
                        [
                            abs(
                                float(self.config.target_velocity_x)
                                - float(reward_body_velocity[0])
                            )
                        ],
                        dtype=np.float32,
                    ),
                    "velocity_estimation_error": np.asarray(
                        [
                            np.linalg.norm(
                                reward_body_velocity - estimated_body_velocity
                            )
                        ],
                        dtype=np.float32,
                    ),
                }
            )

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
                observation = self._logical_reset(observation)
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


    def close(self):
        self.client.close()
