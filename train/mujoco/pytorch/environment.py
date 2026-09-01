"""SDK2/MuJoCo implementation of the shared Go2 environment."""
from __future__ import annotations

import logging
import time

import numpy as np

from train.core.base import (
    ACTION_SIZE, ACTION_SPEC, CONTROL_DT, DEFAULT_JOINT_POSITION,
    Go2Environment, InvalidTransitionError, PHYSICS_DT,
    PHYSICS_STEPS_PER_ACTION, ActionMapper,
    format_policy_io_contract,
    validate_environment_contract,
)
from train.core.estimation import (
    VelocityEstimator,
    foot_position_velocity_body, quaternion_rotation_matrix_wxyz,
    velocity_estimator_config_from,
)
from train.core.task import (
    EpisodeTracker, ObservationBuilder, build_observation, compute_reward,
    local_base_clearance,
)

from .config import get_config
from .client import SDKClient
from .fall import FallDetector
from .mjcf import validate_go2_mjcf_contract
from .reset import MujocoResetController
from .state import SIMULATOR_TICK_SECONDS, StateBufferError, StateTimeout


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
        validate_environment_contract(environment)
        if int(environment.nr_envs) != 1:
            raise ValueError(
                "The SDK2/MuJoCo bridge exposes exactly one robot; "
                f"environment.nr_envs must be 1, got {environment.nr_envs}"
            )
        validate_go2_mjcf_contract()
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
        episode_steps = int(environment.episode_steps)
        if role == "eval":
            episode_steps = int(
                getattr(environment, "evaluation_episode_steps", episode_steps)
            )
        if episode_steps < 1:
            raise ValueError("MuJoCo episode length must be positive")
        self.episode = EpisodeTracker(episode_steps)
        self._last_tick: int | None = None
        self._generation = 0
        self._last_observation: np.ndarray | None = None
        self._policy_blend_elapsed = 0.0
        logger.info(
            "\n" + format_policy_io_contract(environment.target_velocity_x)
        )

    def _wait_window(
        self,
        count: int | None = None,
        *,
        expected_command_sequence: int | None = None,
    ):
        count = int(count or self.config.policy_frames)
        previous_tick = self._last_tick
        try:
            frames = self.client.state_buffer.wait_for_frames(
                count=count,
                after_tick=self._last_tick,
                timeout=float(self.config.state_timeout),
                generation=self._generation,
                command_sequence=expected_command_sequence,
            )
        except StateBufferError as exc:
            raise InvalidTransitionError(str(exc)) from exc
        expected_tick_stride = int(round(PHYSICS_DT / SIMULATOR_TICK_SECONDS))
        ticks = np.asarray([int(frame.tick) for frame in frames], dtype=np.int64)
        if (
            previous_tick is not None
            and ticks.size > 0
            and int(ticks[0]) != int(previous_tick) + expected_tick_stride
        ):
            raise InvalidTransitionError(
                "MuJoCo LowState window is missing its first 2 ms physics frame: "
                f"expected tick {int(previous_tick) + expected_tick_stride}, "
                f"received {int(ticks[0])}"
            )
        if ticks.size > 1 and not np.all(np.diff(ticks) == expected_tick_stride):
            raise InvalidTransitionError(
                "MuJoCo LowState window is missing a 2 ms physics frame: "
                f"received ticks {ticks.tolist()}"
            )
        if expected_command_sequence is not None:
            sequences = [
                getattr(frame, "command_sequence", None) for frame in frames
            ]
            if any(
                sequence is None
                or int(sequence) != int(expected_command_sequence)
                for sequence in sequences
            ):
                raise InvalidTransitionError(
                    "MuJoCo LowState window is not an acknowledged post-step "
                    "window for the published LowCmd: expected sequence "
                    f"{int(expected_command_sequence)}, received {sequences}. "
                    "Run the bridge with ORSA_STRICT_LOCKSTEP=1 after applying "
                    "assets/robots/go2/unitree_mujoco_bridge_lockstep.patch."
                )
        self._last_tick = int(frames[-1].tick)
        return frames

    def _assert_lockstep_boundary(self) -> None:
        """Reject state evolution that the replay transition cannot represent."""

        latest_tick = self.client.state_buffer.last_tick
        if (
            self._last_tick is not None
            and latest_tick is not None
            and int(latest_tick) != int(self._last_tick)
        ):
            raise InvalidTransitionError(
                "MuJoCo advanced while the policy/learner was outside env.step: "
                f"the policy observed tick {int(self._last_tick)}, but the "
                f"bridge reached tick {int(latest_tick)} before LowCmd publish. "
                "Strict replay semantics require the patched command-driven "
                "bridge with ORSA_STRICT_LOCKSTEP=1."
            )

    def _training_states_for_frames(self, frames):
        provider = getattr(self.client, "training_states_for_ticks", None)
        if not callable(provider):
            raise InvalidTransitionError(
                "MuJoCo client cannot provide timestamp-synchronized base/root truth"
            )
        ticks = tuple(int(frame.tick) for frame in frames)
        sequences = tuple(int(frame.command_sequence) for frame in frames)
        try:
            truth_frames = provider(
                ticks,
                command_sequences=sequences,
                generation=self._generation,
                timeout=float(self.config.state_timeout),
            )
        except StateBufferError as exc:
            raise InvalidTransitionError(str(exc)) from exc
        if len(truth_frames) != len(frames):
            raise InvalidTransitionError(
                "MuJoCo root truth count does not match the LowState window"
            )
        for tick, sequence, truth in zip(ticks, sequences, truth_frames):
            if int(getattr(truth, "tick", -1)) != tick:
                raise InvalidTransitionError(
                    f"MuJoCo root truth tick {getattr(truth, 'tick', None)} "
                    f"does not match LowState tick {tick}"
                )
            if int(getattr(truth, "command_sequence", -1)) != sequence:
                raise InvalidTransitionError(
                    "MuJoCo root truth command sequence "
                    f"{getattr(truth, 'command_sequence', None)} does not match "
                    f"LowState sequence {sequence} at tick {tick}"
                )
            position = np.asarray(truth.base_position, dtype=np.float64)
            velocity = np.asarray(truth.world_velocity, dtype=np.float64)
            if (
                position.shape != (3,)
                or velocity.shape != (3,)
                or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(velocity))
            ):
                raise InvalidTransitionError(
                    f"MuJoCo root truth at tick {tick} is incomplete or non-finite"
                )
        return truth_frames

    def _reset_pose_diagnostics(self, state) -> dict[str, object]:
        joint_error = np.max(
            np.abs(
                np.asarray(state.joint_q, dtype=np.float32)
                - DEFAULT_JOINT_POSITION
            )
        )
        max_joint_velocity = np.max(
            np.abs(np.asarray(state.joint_dq, dtype=np.float32))
        )
        reset_quaternion = np.asarray(state.imu_quat, dtype=np.float64)
        reset_quaternion_norm = float(np.linalg.norm(reset_quaternion))
        identity_angle = float("inf")
        orientation_ready = False
        if np.isfinite(reset_quaternion_norm) and reset_quaternion_norm > 1.0e-8:
            identity_angle = 2.0 * np.arccos(
                np.clip(
                    abs(float(reset_quaternion[0])) / reset_quaternion_norm,
                    0.0,
                    1.0,
                )
            )
            orientation_ready = identity_angle <= float(
                self.config.reset_angle_tolerance
            )
        joints_ready = joint_error <= float(self.config.reset_joint_tolerance)
        joints_still = max_joint_velocity <= float(
            self.config.reset_max_joint_velocity
        )

        truth = self._training_states_for_frames([state])[0]
        position = np.asarray(truth.base_position, dtype=np.float32)
        height_ready = abs(
            float(position[2]) - float(self.config.reset_base_height)
        ) <= float(self.config.reset_base_height_tolerance)
        rotation = quaternion_rotation_matrix_wxyz(state.imu_quat)
        foot_position_body, _ = foot_position_velocity_body(
            state.joint_q, state.joint_dq
        )
        foot_position_world = (
            position.astype(np.float64, copy=False)[None, :]
            + (rotation @ foot_position_body.T).T
        )
        foot_surface_height = (
            foot_position_world[:, 2] - float(self.config.foot_collision_radius)
        )
        feet_on_ground = bool(
            np.all(
                np.abs(foot_surface_height)
                <= float(self.config.reset_foot_surface_tolerance)
            )
        )
        ready = (
            orientation_ready
            and joints_ready
            and joints_still
            and height_ready
            and feet_on_ground
        )
        return {
            "ready": ready,
            "joint_error": float(joint_error),
            "max_joint_velocity": float(max_joint_velocity),
            "identity_angle": float(identity_angle),
            "base_height": float(position[2]),
            "base_height_error": abs(
                float(position[2]) - float(self.config.reset_base_height)
            ),
            "foot_surface_height": foot_surface_height,
            "orientation_ready": orientation_ready,
            "joints_ready": bool(joints_ready),
            "joints_still": bool(joints_still),
            "height_ready": bool(height_ready),
            "feet_on_ground": feet_on_ground,
        }

    def _reset_pose_ready(self, state) -> bool:
        return bool(self._reset_pose_diagnostics(state)["ready"])

    def _log_reset_pose_not_ready(self, state, stage: str) -> None:
        diagnostics = self._reset_pose_diagnostics(state)
        failed = [
            name
            for name in (
                "orientation_ready",
                "joints_ready",
                "joints_still",
                "height_ready",
                "feet_on_ground",
            )
            if not diagnostics[name]
        ]
        foot_heights = np.asarray(
            diagnostics["foot_surface_height"], dtype=np.float64
        )
        logger.warning(
            "Reset pose not ready after %s; failed=%s, joint_error=%.5f "
            "(limit %.5f), max_joint_velocity=%.5f (limit %.5f), "
            "identity_angle=%.5f (limit %.5f), base_height=%.5f, "
            "height_error=%.5f (limit %.5f), foot_surface_height=%s "
            "(abs limit %.5f)",
            stage,
            ",".join(failed),
            diagnostics["joint_error"],
            float(self.config.reset_joint_tolerance),
            diagnostics["max_joint_velocity"],
            float(self.config.reset_max_joint_velocity),
            diagnostics["identity_angle"],
            float(self.config.reset_angle_tolerance),
            diagnostics["base_height"],
            diagnostics["base_height_error"],
            float(self.config.reset_base_height_tolerance),
            np.array2string(foot_heights, precision=5, separator=","),
            float(self.config.reset_foot_surface_tolerance),
        )

    def _duration_steps(self, seconds: float) -> int:
        period = float(self.config.policy_period_seconds)
        if period <= 0.0:
            raise ValueError("environment.policy_period_seconds must be positive")
        return max(0, int(np.ceil(float(seconds) / period)))

    def _publish_and_wait(self, target: np.ndarray):
        self._assert_lockstep_boundary()
        target = np.asarray(target, dtype=np.float32)
        if isinstance(self.client, SDKClient):
            command_sequence = self.client.publish_joint_target(
                target,
                kp=float(self.config.reset_kp),
                kd=float(self.config.reset_kd),
            )
        else:
            self.client.publish_joint_target(target)
            command_sequence = None
        if command_sequence is None:
            return self._wait_window()[-1]
        return self._wait_window(
            expected_command_sequence=command_sequence
        )[-1]

    def _wait_for_simulator_restart(self, reason: str) -> None:
        logger.warning(
            f"{reason} Press Backspace in the unitree_mujoco window to reset."
        )
        timeout = float(self.config.manual_reset_timeout)
        timeout = None if timeout < 0 else timeout
        arm_restart = getattr(self.client, "arm_simulator_restart", None)
        if callable(arm_restart):
            arm_restart()
        else:
            self.client.state_buffer.arm_restart()
        try:
            self._generation = self.client.state_buffer.wait_for_restart(
                self._generation, timeout=timeout
            )
        except Exception:
            cancel_restart = getattr(self.client, "cancel_simulator_restart", None)
            if callable(cancel_restart):
                cancel_restart()
            else:
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

        # Arm the SDK bridge before resetting physics. LowCmd is retained by
        # unitree_mujoco while mjData is reset. The bundled MJCF makes ordinary
        # qpos0 identical to the canonical home pose, so named-key reset support
        # is neither assumed nor required.
        if isinstance(self.client, SDKClient):
            # Complete and acknowledge the final command transaction before
            # arming rollback detection. Otherwise reset could race the DDS
            # subscriber and make sequence N integrate after mj_resetData.
            self._publish_and_wait(
                DEFAULT_JOINT_POSITION,
            )
        else:
            self.client.publish_joint_target(DEFAULT_JOINT_POSITION)

        attempts = max(1, int(self.config.auto_reset_attempts))
        last_timeout = None
        for attempt in range(1, attempts + 1):
            generation = self.client.state_buffer.generation
            arm_restart = getattr(self.client, "arm_simulator_restart", None)
            if callable(arm_restart):
                arm_restart()
            else:
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
                cancel_restart = getattr(
                    self.client, "cancel_simulator_restart", None
                )
                if callable(cancel_restart):
                    cancel_restart()
                else:
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
        """Verify a stable standing reset, with interpolation only as fallback."""

        state = initial_state
        # A normal reset now loads canonical qpos0. Return the first verified
        # feedback frame so reset observations have the same instantaneous
        # semantics as Isaac rather than a state evolved through a 2 s hold.
        for _ in range(
            max(1, self._duration_steps(self.config.reset_sync_timeout_seconds))
        ):
            if self._reset_pose_ready(state):
                return state
            state = self._publish_and_wait(DEFAULT_JOINT_POSITION)

        self._log_reset_pose_not_ready(state, "initial reset settling")

        # Recovery fallback for an external simulator that was not reset or a
        # genuinely fallen state. Readiness is measured, never assumed.
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
            for _ in range(
                self._duration_steps(self.config.reset_sync_timeout_seconds)
            ):
                if self._reset_pose_ready(state):
                    return state
                state = self._publish_and_wait(DEFAULT_JOINT_POSITION)
            if self._reset_pose_ready(state):
                return state

            self._log_reset_pose_not_ready(state, "stand-up interpolation")

            self._auto_reset_simulator(
                "Go2 did not reach the standing pose after interpolation."
            )
            state = self.client.state_buffer.latest_state
            if state is None:
                state = self._wait_window(count=1)[-1]

    def _finish_physical_reset(self):
        self.action_mapper.reset()
        self.observation_builder.reset()
        self.fall_detector.reset()
        self.episode.reset()
        self._policy_blend_elapsed = 0.0
        # Capture the measured reset pose before issuing any command. Stand-up
        # interpolation begins from this feedback, not from an assumed pose.
        state = self.client.state_buffer.latest_state
        if state is None:
            self._last_tick = None
            state = self._wait_window(count=1)[-1]
        else:
            self._last_tick = int(state.tick)
        state = self._hold_reset_pose(state)
        # Reset observations use the estimator prior, exactly as Isaac: the
        # first feedback sample establishes quaternion continuity but does not
        # integrate a backend-specific extra 2 ms acceleration update.
        observation, quaternion = build_observation(
            state,
            np.zeros(3, dtype=np.float32),
            self.observation_builder.previous_q_target,
        )
        self.observation_builder.previous_quaternion = quaternion
        self._last_observation = observation
        return observation[None, :], {}

    def _physical_episode_reset(
        self,
        reason: str,
        delay_seconds: float = 0.0,
        *,
        automatic: bool = True,
    ):
        if automatic and self.reset_controller is not None:
            self._auto_reset_simulator(
                reason,
                delay_seconds=delay_seconds,
            )
        else:
            self._wait_for_simulator_restart(reason)
        return self._finish_physical_reset()[0][0]

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.client.start()
        self._generation = self.client.state_buffer.generation
        # Observe a pre-reset tick so StateBuffer can recognize the rollback
        # caused by ordinary MuJoCo reset. This is done for train and eval alike.
        self._last_tick = self.client.state_buffer.last_tick
        if self._last_tick is None:
            self._wait_window(count=1)
        if bool(self.config.auto_reset_on_start) and self.reset_controller is not None:
            self._auto_reset_simulator("Policy environment reset.")
        else:
            self._wait_for_simulator_restart("Policy environment reset requested.")
        return self._finish_physical_reset()

    def _manual_failure_reset(self):
        automatic = bool(self.config.auto_reset_after_fall)
        delay = float(self.config.fall_auto_reset_delay_seconds) if automatic else 0.0
        return self._physical_episode_reset(
            "Go2 fall detected.",
            delay,
            automatic=automatic,
        )

    def step(self, actions):
        action = np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_SIZE)[0]
        self._assert_lockstep_boundary()
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
        if isinstance(self.client, SDKClient):
            command_sequence = self.client.publish_joint_target(
                mapped.q_target,
                kp=float(self.config.policy_kp),
                kd=float(self.config.policy_kd),
            )
        else:
            self.client.publish_joint_target(mapped.q_target)
            command_sequence = None
        if command_sequence is None:
            frames = self._wait_window()
        else:
            frames = self._wait_window(
                expected_command_sequence=command_sequence
            )
        # Only an acknowledged complete action window may become the previous
        # target exposed in the next observation.
        self.observation_builder.set_previous_q_target(mapped.q_target)
        truth_frames = self._training_states_for_frames(frames)

        failure = False
        tilt_failure = False
        height_failure = False
        transition_index = len(frames) - 1
        for index, (frame, frame_truth) in enumerate(zip(frames, truth_frames)):
            frame_clearance = float(
                local_base_clearance(
                    np.asarray(frame_truth.base_position, dtype=np.float64)[2],
                    0.0,
                )
            )
            frame_tilt_failure = self.fall_detector.update(frame.imu_quat)
            frame_height_failure = self.fall_detector.update_base_clearance(
                frame_clearance
            )
            if frame_tilt_failure or frame_height_failure:
                transition_index = index
                tilt_failure = frame_tilt_failure
                height_failure = frame_height_failure
                failure = True
                break
        final_state = frames[transition_index]
        truth = truth_frames[transition_index]
        observation, estimated_body_velocity = self.observation_builder.build_many(
            frames[: transition_index + 1]
        )

        if final_state.actuator_torque is None:
            raise InvalidTransitionError("MuJoCo reward requires joint torque feedback")
        world_velocity = np.asarray(truth.world_velocity)
        rotation = quaternion_rotation_matrix_wxyz(final_state.imu_quat)
        reward_body_velocity = rotation.T @ np.asarray(world_velocity)
        # The canonical SDK2 MuJoCo scene has a zero-height ground plane,
        # so world z is exactly the local terrain clearance.
        base_clearance = float(
            local_base_clearance(np.asarray(truth.base_position)[2], 0.0)
        )
        foot_position_body, _ = foot_position_velocity_body(
            final_state.joint_q, final_state.joint_dq
        )
        foot_position_world = (
            np.asarray(truth.base_position, dtype=np.float64)[None, :]
            + (rotation @ foot_position_body.T).T
        )
        foot_clearance = foot_position_world[:, 2]
        terms = compute_reward(
            reward_body_velocity,
            final_state.imu_quat,
            final_state.imu_gyro,
            final_state.actuator_torque,
            float(self.config.target_velocity_x),
        )
        terminated, truncated = self.episode.advance(terms.total, failure)
        torque_saturation_ratio = float(
            np.mean(
                np.abs(np.asarray(final_state.actuator_torque))
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
                [np.mean(np.abs(mapped.applied_action) > 0.98)], dtype=np.float32
            ),
            "torque_saturation_ratio": np.asarray(
                [torque_saturation_ratio], dtype=np.float32
            ),
            "estimated_forward_velocity": np.asarray(
                [estimated_body_velocity[0]], dtype=np.float32
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
                observation = self._physical_episode_reset(
                    "Episode time limit reached."
                )
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
