"""RL-X adapter around Isaac Lab's manager-based Go2 environment."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch
from gymnasium.spaces import Box
from rl_x.environments.safety_rollout import RolloutStep

from ..common.specs import (
    ACTION_SPEC,
    ACTION_SIZE,
    DEFAULT_JOINT_POSITION,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    OBSERVATION_SPEC,
    OBSERVATION_SIZE,
    PHYSICS_STEPS_PER_ACTION,
    format_policy_io_contract,
)
from ..common.manifest import (
    build_manifest,
    validate_manifest,
    validate_transfer_manifest,
)
from ..common.reward import (
    BASE_HEIGHT_TARGET,
    FOOT_CLEARANCE_TARGET,
    PHASE_EPSILON,
    PHASE_REFERENCE_FREQUENCY,
    REWARD_DT,
    REWARD_DEFAULT_JOINT_POSITION,
    REWARD_SCALES,
    SWING_SPEED_FULL,
    SWING_SPEED_START,
    TRACKING_SIGMA,
    add_terminal_failure_penalty,
    local_base_clearance,
)
from ..common.estimation import (
    TorchVelocityEstimator,
    velocity_estimator_config_from,
)
from .general_properties import GeneralProperties
from .mdp import (
    build_observation_tensor,
    default_joint_target,
    sdk_joint_indices,
    validate_action_term_contract,
)
from .randomization_cfg import TRANSFER_RANDOMIZATION


def relabel_isaac_backend_manager_output(output: str) -> str:
    """Make clear which Isaac manager outputs the Go2 adapter discards."""

    replacements = (
        (
            "[INFO] Command Manager:",
            "[INFO] IsaacLab backend Command Manager "
            "(internal; not the Go2 reward target):",
        ),
        (
            "[INFO] Observation Manager:",
            "[INFO] IsaacLab backend Observation Manager "
            "(internal tensor discarded by the Go2 RL-X adapter):",
        ),
        (
            "Active Observation Terms in Group: 'policy'",
            "Active internal backend terms [not model input] in Group: 'policy'",
        ),
        (
            "[INFO] Reward Manager:",
            "[INFO] IsaacLab backend Reward Manager "
            "(disabled; Go2 adapter computes the effective reward):",
        ),
    )
    for original, replacement in replacements:
        output = output.replace(original, replacement)
    return output


def project_action_targets_tensor(previous_q_target, actions):
    """Torch-native, differentiable counterpart of the common projection."""

    if not torch.is_tensor(actions):
        actions = torch.as_tensor(actions, dtype=torch.float32)
    if not torch.is_tensor(previous_q_target):
        previous_q_target = torch.as_tensor(
            previous_q_target, dtype=actions.dtype, device=actions.device
        )
    else:
        previous_q_target = previous_q_target.to(
            dtype=actions.dtype, device=actions.device
        )
    default = torch.as_tensor(
        DEFAULT_JOINT_POSITION, dtype=actions.dtype, device=actions.device
    )
    scale = torch.as_tensor(
        ACTION_SPEC.scale, dtype=actions.dtype, device=actions.device
    )
    lower = torch.as_tensor(
        JOINT_LOWER_LIMIT, dtype=actions.dtype, device=actions.device
    )
    upper = torch.as_tensor(
        JOINT_UPPER_LIMIT, dtype=actions.dtype, device=actions.device
    )
    raw_target = default + scale * actions.clamp(-1, 1)
    raw_target = raw_target.clamp(lower, upper)
    max_delta = ACTION_SPEC.max_target_rate * ACTION_SPEC.control_dt
    target = torch.maximum(
        torch.minimum(raw_target, previous_q_target + max_delta),
        previous_q_target - max_delta,
    )
    applied = ((target - default) / scale).clamp(-1, 1)
    return applied, target


def swing_clearance_and_phase_tensor(
    foot_clearance,
    foot_velocity,
    *,
    clearance_target=FOOT_CLEARANCE_TARGET,
    reference_frequency=PHASE_REFERENCE_FREQUENCY,
    clearance_aggregation="swing_weighted",
):
    """Torch counterpart of the common contact-free gait reward terms."""

    horizontal_speed = torch.linalg.vector_norm(foot_velocity[..., :2], dim=-1)
    swing_weight = (
        (horizontal_speed - SWING_SPEED_START)
        / (SWING_SPEED_FULL - SWING_SPEED_START)
    ).clamp(0.0, 1.0)
    deficit = (float(clearance_target) - foot_clearance).clamp_min(0.0)
    if clearance_aggregation == "legacy_mean":
        clearance_error = (swing_weight * deficit.square()).mean(dim=-1)
    elif clearance_aggregation == "swing_weighted":
        clearance_error = (swing_weight * deficit.square()).sum(dim=-1) / (
            swing_weight.sum(dim=-1).clamp_min(1.0)
        )
    else:
        raise ValueError(
            "clearance_reward_mode must be 'legacy_mean' or 'swing_weighted'"
        )

    x = (
        foot_clearance.clamp(0.0, float(clearance_target))
        - 0.5 * float(clearance_target)
    ) / (0.5 * float(clearance_target))
    y = foot_velocity[..., 2] / (
        torch.pi * float(reference_frequency) * float(clearance_target)
    )
    phase = torch.stack((x, y), dim=-1)
    phase = phase / torch.linalg.vector_norm(
        phase, dim=-1, keepdim=True
    ).clamp_min(PHASE_EPSILON)
    # Shared SDK order is FR, FL, RR, RL.
    fr, fl, rr, rl = phase.unbind(dim=-2)
    phase_score = 0.25 * (
        0.5 * (1.0 + (fr * rl).sum(dim=-1))
        + 0.5 * (1.0 + (fl * rr).sum(dim=-1))
        + 0.5 * (1.0 - (fr * fl).sum(dim=-1))
        + 0.5 * (1.0 - (rl * rr).sum(dim=-1))
    )
    swing_activity = (swing_weight.sum(dim=-1) / 2.0).clamp(0.0, 1.0)
    return clearance_error, swing_activity * phase_score, swing_weight


def swing_clearance_overshoot_tensor(
    foot_clearance,
    swing_weight,
    *,
    upper_target,
):
    overshoot = (foot_clearance - float(upper_target)).clamp_min(0.0)
    return (swing_weight * overshoot.square()).sum(dim=-1) / (
        swing_weight.sum(dim=-1).clamp_min(1.0)
    )


def movement_reward_gate_tensor(forward_velocity, *, start, full):
    start = float(start)
    full = float(full)
    if full <= start:
        return torch.ones_like(forward_velocity)
    return ((forward_velocity - start) / (full - start)).clamp(0.0, 1.0)


class TorchFallDetector:
    """Vectorized common tilt-or-low-base SQRL incident detector."""

    def __init__(
        self,
        nr_envs: int,
        device,
        angle_threshold: float = 0.8,
        min_base_clearance: float = 0.18,
        consecutive_frames: int = 5,
        samples_per_update: int = 1,
    ):
        if consecutive_frames < 1:
            raise ValueError("consecutive_frames must be at least 1")
        if samples_per_update < 1:
            raise ValueError("samples_per_update must be at least 1")
        self.angle_threshold = float(angle_threshold)
        self.min_base_clearance = float(min_base_clearance)
        self.consecutive_frames = int(consecutive_frames)
        self.samples_per_update = int(samples_per_update)
        self.tilt_count = torch.zeros(nr_envs, dtype=torch.long, device=device)
        self.height_count = torch.zeros(nr_envs, dtype=torch.long, device=device)
        self.last_tilt_failure = torch.zeros(
            nr_envs, dtype=torch.bool, device=device
        )
        self.last_height_failure = torch.zeros(
            nr_envs, dtype=torch.bool, device=device
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            self.tilt_count.zero_()
            self.height_count.zero_()
            self.last_tilt_failure.zero_()
            self.last_height_failure.zero_()
        else:
            self.tilt_count[env_ids] = 0
            self.height_count[env_ids] = 0
            self.last_tilt_failure[env_ids] = False
            self.last_height_failure[env_ids] = False

    @torch.no_grad()
    def update(self, quaternion, base_clearance=None):
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        w, x, y, z = quaternion.unbind(dim=-1)
        roll = torch.atan2(
            2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square())
        )
        pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
        tilted = (roll.abs() > self.angle_threshold) | (
            pitch.abs() > self.angle_threshold
        )
        self.tilt_count = torch.where(
            tilted,
            self.tilt_count + self.samples_per_update,
            torch.zeros_like(self.tilt_count),
        )
        if base_clearance is None:
            low = torch.zeros_like(tilted)
        else:
            low = torch.as_tensor(
                base_clearance,
                dtype=quaternion.dtype,
                device=quaternion.device,
            ).reshape(tilted.shape) < self.min_base_clearance
        self.height_count = torch.where(
            low,
            self.height_count + self.samples_per_update,
            torch.zeros_like(self.height_count),
        )
        self.last_tilt_failure = self.tilt_count >= self.consecutive_frames
        self.last_height_failure = self.height_count >= self.consecutive_frames
        return self.last_tilt_failure | self.last_height_failure


class Go2IsaacEnv:
    general_properties = GeneralProperties

    def __init__(self, config, backend=None):
        self._owns_backend = backend is None
        if backend is None:
            from isaaclab.envs import ManagerBasedRLEnv
            from .env_cfg import make_env_cfg
            from .randomization_cfg import format_domain_randomization_report

            class ReportedManagerBasedRLEnv(ManagerBasedRLEnv):
                def load_managers(self):
                    captured = StringIO()
                    try:
                        with redirect_stdout(captured):
                            super().load_managers()
                    finally:
                        output = relabel_isaac_backend_manager_output(
                            captured.getvalue()
                        )
                        print(
                            output,
                            end="" if output.endswith("\n") else "\n",
                            flush=True,
                        )

            backend = ReportedManagerBasedRLEnv(cfg=make_env_cfg(config))
            print(
                format_domain_randomization_report(
                    enabled=bool(config.environment.domain_randomization),
                    friction=float(config.environment.friction),
                ),
                flush=True,
            )
        self.backend = backend
        self.config = config.environment
        self._select_playback_terrain()
        self._domain_randomization = bool(self.config.domain_randomization)
        self.nr_envs = int(self.config.nr_envs)
        self.num_envs = self.nr_envs
        self.nr_task_envs = int(self.config.nr_task_envs)
        self.nr_safety_envs = int(self.config.nr_safety_envs)
        if self.nr_task_envs + self.nr_safety_envs != self.nr_envs:
            raise ValueError("Isaac task and safety pool sizes must sum to nr_envs")
        self.single_observation_space = Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.single_action_space = Box(
            low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.policy_observation_indices = np.arange(OBSERVATION_SIZE)
        self.critic_observation_indices = np.arange(OBSERVATION_SIZE)
        self.safety_critic_observation_indices = np.arange(OBSERVATION_SIZE)
        robot = self.backend.scene["robot"]
        validate_action_term_contract(
            self.backend.action_manager.get_term("joint_pos")
        )
        self._joint_indices = sdk_joint_indices(robot.joint_names, robot.device)
        foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        try:
            foot_indices = [robot.body_names.index(name) for name in foot_names]
        except ValueError as error:
            raise RuntimeError(
                f"Expected Go2 SDK-order foot bodies {foot_names}, got "
                f"{robot.body_names}"
            ) from error
        self._foot_body_indices = torch.as_tensor(
            foot_indices, dtype=torch.long, device=robot.device
        )
        self._previous_target = default_joint_target(self.nr_envs, robot.device)
        self._previous_reward_action = torch.zeros(
            (self.nr_envs, ACTION_SIZE), device=robot.device
        )
        self._previous_quaternion = None
        self._velocity_estimator = TorchVelocityEstimator(
            self.nr_envs,
            robot.device,
            dt=ACTION_SPEC.control_dt,
            config=velocity_estimator_config_from(self.config),
        )
        self._fall_detector = TorchFallDetector(
            self.nr_envs,
            robot.device,
            angle_threshold=float(self.config.fall_angle_threshold),
            min_base_clearance=float(self.config.fall_min_base_clearance),
            consecutive_frames=int(self.config.fall_consecutive_frames),
            # The adapter observes orientation once after each decimated policy
            # step.  Treating that sample as the latest 10 physics frames is a
            # conservative discrete approximation of SDK's 500 Hz sustained
            # detector and keeps the manifest's frame unit identical.
            samples_per_update=PHYSICS_STEPS_PER_ACTION,
        )
        self._episode_return = torch.zeros(self.nr_envs, device=robot.device)
        self._episode_length = torch.zeros(
            self.nr_envs, device=robot.device, dtype=torch.long
        )
        self._episode_start_x = robot.data.root_pos_w[:, 0].clone()
        self._episode_max_progress = torch.zeros(self.nr_envs, device=robot.device)
        self._velocity_history = torch.zeros(
            (self.nr_envs, 100), device=robot.device
        )
        self._velocity_history_count = torch.zeros(
            self.nr_envs, dtype=torch.long, device=robot.device
        )
        self._velocity_history_cursor = 0
        print(
            format_policy_io_contract(self.config.target_velocity_x),
            flush=True,
        )

    def _select_playback_terrain(self):
        """Place all playback envs on the requested generated terrain family."""

        from .terrain_cfg import playback_terrain_column

        terrain_type = str(self.config.playback_terrain_type)
        terrain_level = int(self.config.playback_terrain_level)
        if terrain_level < -1:
            raise ValueError("playback_terrain_level must be -1 or non-negative")
        terrain = self.backend.scene.terrain
        terrain_origins = getattr(terrain, "terrain_origins", None)
        if terrain_type.lower() == "auto" and terrain_level < 0:
            return
        if terrain_origins is None:
            raise ValueError(
                "playback terrain selection requires environment.terrain_mode=rough"
            )
        column = playback_terrain_column(terrain_type, terrain_origins.shape[1])
        if column is not None:
            terrain.terrain_types.fill_(column)
        if terrain_level >= terrain_origins.shape[0]:
            raise ValueError(
                f"playback_terrain_level must be -1 or less than "
                f"{terrain_origins.shape[0]}, got {terrain_level}"
            )
        if terrain_level >= 0:
            terrain.terrain_levels.fill_(terrain_level)
        terrain.env_origins[:] = terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        print(
            f"[INFO] Playback terrain: {terrain_type.lower()} "
            f"(column {int(terrain.terrain_types[0])}/"
            f"{terrain_origins.shape[1] - 1}, level "
            f"{int(terrain.terrain_levels[0])}/{terrain_origins.shape[0] - 1})",
            flush=True,
        )

    @staticmethod
    def project_actions(states, actions):
        """Return side-effect-free applied actions in the executable domain."""

        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        if not torch.is_tensor(states):
            states = torch.as_tensor(
                states, dtype=actions.dtype, device=actions.device
            )
        else:
            states = states.to(dtype=actions.dtype, device=actions.device)
        if states.shape[-1] != OBSERVATION_SIZE:
            raise ValueError(f"Observation must end in 46 values, got {states.shape}")
        applied, _ = project_action_targets_tensor(
            states[..., OBSERVATION_SPEC.previous_action_q_target], actions
        )
        return applied

    @property
    def _robot(self):
        return self.backend.scene["robot"]

    def _sensor_value(self, value, noise_name):
        value = value.clone()
        if not self._domain_randomization:
            return value
        low, high = TRANSFER_RANDOMIZATION[noise_name]
        return value + torch.empty_like(value).uniform_(float(low), float(high))

    def _observation(self):
        robot = self._robot
        imu = self.backend.scene["imu"]
        joint_q = self._sensor_value(
            robot.data.joint_pos[:, self._joint_indices], "joint_position_noise"
        )
        joint_dq = self._sensor_value(
            robot.data.joint_vel[:, self._joint_indices], "joint_velocity_noise"
        )
        imu_gyro = self._sensor_value(imu.data.ang_vel_b, "gyro_noise")
        imu_acceleration = self._sensor_value(
            imu.data.lin_acc_b, "accelerometer_noise"
        )
        self._latest_estimated_body_velocity = self._velocity_estimator.update(
            joint_q,
            joint_dq,
            imu_gyro,
            imu.data.quat_w,
            imu_acceleration,
        )
        observation, quaternion = build_observation_tensor(
            joint_q,
            joint_dq,
            imu_gyro,
            self._latest_estimated_body_velocity,
            imu.data.quat_w,
            self._previous_target,
            self._previous_quaternion,
        )
        self._previous_quaternion = quaternion
        return observation

    def _reset_done_state(self, done):
        env_ids = torch.nonzero(done).flatten()
        if env_ids.numel() == 0:
            return
        self._previous_target[env_ids] = torch.as_tensor(
            DEFAULT_JOINT_POSITION,
            dtype=self._previous_target.dtype,
            device=self._previous_target.device,
        )
        self._previous_reward_action[env_ids] = 0.0
        self._velocity_estimator.reset(env_ids)
        self._fall_detector.reset(env_ids)
        if self._previous_quaternion is not None:
            current = self.backend.scene["imu"].data.quat_w[env_ids].clone()
            current = current / torch.linalg.vector_norm(
                current, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            current = torch.where(current[:, :1] < 0, -current, current)
            self._previous_quaternion[env_ids] = current

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.backend.reset()
        self._previous_target = default_joint_target(
            self.nr_envs, self._robot.device
        )
        self._previous_reward_action.zero_()
        self._previous_quaternion = None
        self._velocity_estimator.reset()
        self._fall_detector.reset()
        self._episode_return.zero_()
        self._episode_length.zero_()
        self._episode_start_x.copy_(self._robot.data.root_pos_w[:, 0])
        self._episode_max_progress.zero_()
        self._velocity_history.zero_()
        self._velocity_history_count.zero_()
        self._velocity_history_cursor = 0
        return self._observation().detach().cpu().numpy(), {}

    def _reset_failed_backend_envs(self, failure, already_reset):
        """Reset only IMU-failed envs not already reset by Isaac's time limit."""

        env_ids = torch.nonzero(failure & ~already_reset).flatten()
        if env_ids.numel() == 0:
            return
        reset_idx = getattr(self.backend, "_reset_idx", None)
        if reset_idx is None:
            raise RuntimeError(
                "Isaac backend does not support indexed reset required by the "
                "adapter IMU failure contract"
            )
        reset_idx(env_ids)

    def step(self, actions):
        action = torch.as_tensor(
            actions, dtype=torch.float32, device=self._robot.device
        ).reshape(self.nr_envs, ACTION_SIZE)
        applied_action, target = project_action_targets_tensor(
            self._previous_target, action
        )
        reward_action = action.clamp(-1.0, 1.0)
        self._previous_target = target

        _, _, backend_terminated, truncated, extras = self.backend.step(applied_action)
        # Clone manager buffers before an adapter-driven indexed reset mutates
        # manager state in place.
        backend_terminated = backend_terminated.clone()
        truncated = truncated.clone()
        robot = self._robot
        del extras

        # Capture all training-only truth before resetting IMU-failed bodies.
        # The policy observes the proprioceptive estimator, not this simulator
        # value, so retain a snapshot for sim-to-sim diagnostics.
        target_velocity = float(self.config.target_velocity_x)
        body_velocity = robot.data.root_lin_vel_b.clone()
        tracking_lin_vel = torch.exp(
            -(
                (target_velocity - body_velocity[:, 0]).square()
                + body_velocity[:, 1].square()
            )
            / TRACKING_SIGMA
        )
        velocity_error = (
            (target_velocity - body_velocity[:, 0]).square()
            + body_velocity[:, 1].square()
        )
        tracking_ang_vel = torch.exp(
            -robot.data.root_ang_vel_b[:, 2].square() / TRACKING_SIGMA
        )
        lin_vel_z = body_velocity[:, 2].square()
        terrain_hits = torch.nan_to_num(
            self.backend.scene["height_scanner"].data.ray_hits_w,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        base_xy = robot.data.root_pos_w[:, None, :2]
        base_hit_distance = (terrain_hits[..., :2] - base_xy).square().sum(dim=-1)
        base_hit_count = min(4, terrain_hits.shape[1])
        base_hit_indices = torch.topk(
            base_hit_distance, k=base_hit_count, largest=False
        ).indices
        terrain_height = torch.gather(
            terrain_hits[..., 2], 1, base_hit_indices
        ).mean(dim=1)
        base_clearance = local_base_clearance(
            robot.data.root_pos_w[:, 2], terrain_height
        )
        base_height_error = (base_clearance - BASE_HEIGHT_TARGET).square()
        foot_position = robot.data.body_pos_w[:, self._foot_body_indices]
        foot_velocity = robot.data.body_lin_vel_w[:, self._foot_body_indices]
        foot_hit_distance = (
            foot_position[..., None, :2]
            - terrain_hits[:, None, :, :2]
        ).square().sum(dim=-1)
        foot_hit_indices = foot_hit_distance.argmin(dim=-1)
        foot_ground_height = torch.gather(
            terrain_hits[..., 2], 1, foot_hit_indices
        )
        foot_clearance = foot_position[..., 2] - foot_ground_height
        foot_clearance_error, phase_reward, swing_weight = (
            swing_clearance_and_phase_tensor(
                foot_clearance,
                foot_velocity,
                clearance_target=float(self.config.foot_clearance_target),
                reference_frequency=float(
                    self.config.phase_reference_frequency
                ),
                clearance_aggregation=str(self.config.clearance_reward_mode),
            )
        )
        foot_clearance_overshoot_error = swing_clearance_overshoot_tensor(
            foot_clearance,
            swing_weight,
            upper_target=float(self.config.foot_clearance_upper_target),
        )
        movement_gate = movement_reward_gate_tensor(
            body_velocity[:, 0],
            start=float(self.config.phase_velocity_gate_start),
            full=float(self.config.phase_velocity_gate_full),
        )
        action_rate = (
            reward_action - self._previous_reward_action
        ).square().sum(dim=-1)
        joint_q = robot.data.joint_pos[:, self._joint_indices]
        default_joint_q = torch.as_tensor(
            REWARD_DEFAULT_JOINT_POSITION,
            dtype=joint_q.dtype,
            device=joint_q.device,
        )
        similar_to_default = torch.abs(joint_q - default_joint_q).sum(dim=-1)
        failure = self._fall_detector.update(
            self.backend.scene["imu"].data.quat_w,
            base_clearance,
        )
        failure = failure.clone()
        tilt_failure = self._fall_detector.last_tilt_failure.clone()
        height_failure = self._fall_detector.last_height_failure.clone()
        terminated = backend_terminated | failure
        progress = robot.data.root_pos_w[:, 0] - self._episode_start_x
        self._episode_max_progress = torch.maximum(
            self._episode_max_progress, progress
        )
        stable_progress_reward = (
            (progress >= float(self.config.stable_progress_start))
            & (
                base_clearance
                >= float(self.config.stable_progress_min_base_clearance)
            )
            & ~failure
        ).float() * movement_gate
        reward_terms = {
            "tracking_lin_vel": REWARD_DT
            * REWARD_SCALES["tracking_lin_vel"]
            * tracking_lin_vel,
            "velocity_error": REWARD_DT
            * REWARD_SCALES["velocity_error"]
            * velocity_error,
            "tracking_ang_vel": REWARD_DT
            * REWARD_SCALES["tracking_ang_vel"]
            * tracking_ang_vel,
            "lin_vel_z": REWARD_DT * REWARD_SCALES["lin_vel_z"] * lin_vel_z,
            "base_height": REWARD_DT
            * REWARD_SCALES["base_height"]
            * base_height_error,
            "foot_clearance": REWARD_DT
            * REWARD_SCALES["foot_clearance"]
            * foot_clearance_error,
            "foot_clearance_overshoot": REWARD_DT
            * float(self.config.foot_clearance_overshoot_scale)
            * foot_clearance_overshoot_error,
            "phase": REWARD_DT
            * float(self.config.phase_reward_scale)
            * movement_gate
            * phase_reward,
            "stable_progress": REWARD_DT
            * float(self.config.stable_progress_scale)
            * stable_progress_reward,
            "action_rate": REWARD_DT
            * REWARD_SCALES["action_rate"]
            * action_rate,
            "similar_to_default": REWARD_DT
            * REWARD_SCALES["similar_to_default"]
            * similar_to_default,
        }
        reward = torch.stack(tuple(reward_terms.values()), dim=0).sum(dim=0)
        failure_reward = failure.float() * float(
            self.config.terminal_failure_penalty
        )
        reward = add_terminal_failure_penalty(
            reward,
            failure.float(),
            float(self.config.terminal_failure_penalty),
        )
        self._previous_reward_action.copy_(reward_action)
        self._velocity_history[:, self._velocity_history_cursor] = body_velocity[:, 0]
        self._velocity_history_cursor = (self._velocity_history_cursor + 1) % 100
        self._velocity_history_count = (self._velocity_history_count + 1).clamp_max(100)
        last_velocity = self._velocity_history.sum(dim=-1) / (
            self._velocity_history_count.clamp_min(1)
        )
        crossing_success = self._episode_max_progress >= float(
            self.config.step_success_distance
        )
        stuck = (~failure) & (last_velocity < 0.1)
        stable_success = crossing_success & ~failure & (last_velocity >= 0.1)
        swing_weight_sum = swing_weight.sum(dim=-1)
        swing_clearance = (swing_weight * foot_clearance).sum(dim=-1) / (
            swing_weight_sum.clamp_min(1.0)
        )
        swing_clearance = torch.where(
            swing_weight_sum > 0.0,
            swing_clearance,
            torch.full_like(swing_clearance, torch.nan),
        )
        already_reset = backend_terminated | truncated
        self._reset_failed_backend_envs(failure, already_reset)
        done = terminated | truncated
        self._reset_done_state(done)
        observation = self._observation()
        estimated_body_velocity = self._latest_estimated_body_velocity.clone()

        self._episode_return += reward
        self._episode_length += 1
        info = {
            "failure": failure.float().cpu().numpy(),
            "failure/tilt": tilt_failure.float().cpu().numpy(),
            "failure/height": height_failure.float().cpu().numpy(),
            "applied_action": applied_action.detach().cpu().numpy(),
            "forward_velocity": body_velocity[:, 0].detach().cpu().numpy(),
            "estimated_forward_velocity": estimated_body_velocity[
                :, 0
            ].detach().cpu().numpy(),
            "target_velocity_error": (
                body_velocity[:, 0] - target_velocity
            ).abs().detach().cpu().numpy(),
            "base_clearance": base_clearance.detach().cpu().numpy(),
            "local_terrain_height": terrain_height.detach().cpu().numpy(),
            "mean_foot_clearance": foot_clearance.mean(dim=-1).detach().cpu().numpy(),
            "max_foot_clearance": foot_clearance.max(dim=-1).values.detach().cpu().numpy(),
            "swing_weighted_foot_clearance": swing_clearance.detach().cpu().numpy(),
            **{
                f"swing_clearance/{leg}": torch.where(
                    swing_weight[:, index] > 0.0,
                    foot_clearance[:, index],
                    torch.full_like(foot_clearance[:, index], torch.nan),
                ).detach().cpu().numpy()
                for index, leg in enumerate(("fr", "fl", "rr", "rl"))
            },
            "swing_ratio/fr": swing_weight[:, 0].detach().cpu().numpy(),
            "swing_ratio/fl": swing_weight[:, 1].detach().cpu().numpy(),
            "swing_ratio/rr": swing_weight[:, 2].detach().cpu().numpy(),
            "swing_ratio/rl": swing_weight[:, 3].detach().cpu().numpy(),
            "terrain/forward_progress": self._episode_max_progress.detach().cpu().numpy(),
            "terrain/success": crossing_success.float().detach().cpu().numpy(),
            "terrain/stable_success": stable_success.float().detach().cpu().numpy(),
            "terrain/stuck": stuck.float().detach().cpu().numpy(),
            "terrain/last_100_velocity": last_velocity.detach().cpu().numpy(),
            "action_saturation_ratio": (
                reward_action.abs() > 0.98
            ).float().mean(dim=-1).detach().cpu().numpy(),
            "torque_saturation_ratio": (
                robot.data.applied_torque[:, self._joint_indices].abs()
                > 0.95 * ACTION_SPEC.effort_limit
            ).float().mean(dim=-1).detach().cpu().numpy(),
            "velocity_estimation_error": torch.linalg.vector_norm(
                estimated_body_velocity - body_velocity, dim=-1
            ).detach().cpu().numpy(),
            **{
                f"reward/{name}": value.detach().cpu().numpy()
                for name, value in reward_terms.items()
            },
            "reward/failure": failure_reward.detach().cpu().numpy(),
            "reward/total": reward.detach().cpu().numpy(),
            "final_observation": [None] * self.nr_envs,
            "final_info": [None] * self.nr_envs,
        }
        for index in torch.nonzero(done).flatten().tolist():
            info["final_observation"][index] = observation[index].detach().cpu().numpy()
            info["final_info"][index] = {
                "episode_return": float(self._episode_return[index]),
                "episode_length": int(self._episode_length[index]),
                "fall": bool(failure[index]),
                "success": bool(crossing_success[index]),
                "stable_success": bool(stable_success[index]),
                "stuck": bool(stuck[index]),
                "forward_progress": float(self._episode_max_progress[index]),
                "last_100_velocity": float(last_velocity[index]),
            }
            self._episode_return[index] = 0
            self._episode_length[index] = 0
            self._episode_start_x[index] = self._robot.data.root_pos_w[index, 0]
            self._episode_max_progress[index] = 0.0
            self._velocity_history[index] = 0.0
            self._velocity_history_count[index] = 0
        return (
            observation.detach().cpu().numpy(),
            reward.detach().cpu().numpy(),
            terminated.detach().cpu().numpy(),
            truncated.detach().cpu().numpy(),
            info,
        )

    def reset_partitions(self):
        observation, _ = self.reset()
        return (
            observation[: self.nr_task_envs],
            observation[self.nr_task_envs :],
        )

    def step_partitions(self, task_actions, safety_actions):
        actions = np.concatenate((task_actions, safety_actions), axis=0)
        observation, reward, terminated, truncated, info = self.step(actions)

        def slice_info(start, stop):
            sliced = {}
            for key, value in info.items():
                if isinstance(value, list):
                    sliced[key] = value[start:stop]
                else:
                    array = np.asarray(value)
                    sliced[key] = array[start:stop]
            return sliced

        split = self.nr_task_envs
        task_step = RolloutStep(
            observation[:split],
            reward[:split],
            terminated[:split],
            truncated[:split],
            slice_info(0, split),
        )
        safety_step = RolloutStep(
            observation[split:],
            reward[split:],
            terminated[split:],
            truncated[split:],
            slice_info(split, self.nr_envs),
        )
        return task_step, safety_step

    def get_final_observation_at_index(self, info, index):
        return info["final_observation"][index]

    def get_final_info_value_at_index(self, info, key, index):
        return info["final_info"][index][key]

    def get_logging_info_dict(self, info):
        ignored = {"failure", "applied_action", "final_observation", "final_info"}
        return {
            key: np.asarray(value).reshape(-1).tolist()
            for key, value in info.items()
            if key not in ignored and not isinstance(value, list)
        }

    def close(self):
        if self._owns_backend:
            self.backend.close()

    def checkpoint_manifest(self, normalizer=None):
        return build_manifest(
            normalizer,
            fall_angle_threshold=float(self.config.fall_angle_threshold),
            fall_min_base_clearance=float(self.config.fall_min_base_clearance),
            fall_consecutive_frames=int(self.config.fall_consecutive_frames),
            target_velocity_x=float(self.config.target_velocity_x),
            foot_clearance_target=float(self.config.foot_clearance_target),
            phase_reference_frequency=float(
                self.config.phase_reference_frequency
            ),
            phase_reward_scale=float(self.config.phase_reward_scale),
            clearance_reward_mode=str(self.config.clearance_reward_mode),
            foot_clearance_upper_target=float(
                self.config.foot_clearance_upper_target
            ),
            foot_clearance_overshoot_scale=float(
                self.config.foot_clearance_overshoot_scale
            ),
            phase_velocity_gate_start=float(
                self.config.phase_velocity_gate_start
            ),
            phase_velocity_gate_full=float(
                self.config.phase_velocity_gate_full
            ),
            stable_progress_start=float(self.config.stable_progress_start),
            stable_progress_min_base_clearance=float(
                self.config.stable_progress_min_base_clearance
            ),
            stable_progress_scale=float(self.config.stable_progress_scale),
            terminal_failure_penalty=float(
                self.config.terminal_failure_penalty
            ),
        )

    def validate_checkpoint_manifest(self, manifest, normalizer=None):
        validate_manifest(manifest, self.checkpoint_manifest(normalizer))

    def validate_transfer_checkpoint_manifest(self, manifest, normalizer=None):
        validate_transfer_manifest(manifest, self.checkpoint_manifest(normalizer))
