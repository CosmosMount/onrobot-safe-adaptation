"""Isaac Lab implementation of the shared Go2 environment."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch

from sqrl.sac.environment import RolloutStep
from train.common.base import (
    ACTION_SIZE, ACTION_SPEC, DEFAULT_JOINT_POSITION, Go2Environment,
    PHYSICS_STEPS_PER_ACTION, format_policy_io_contract,
    project_action_targets_tensor, validate_environment_contract,
)
from train.common.estimation import TorchVelocityEstimator, velocity_estimator_config_from
from .setup import (
    TRANSFER_RANDOMIZATION, build_observation_tensor, default_joint_target,
    format_domain_randomization_report, make_env_cfg,
    playback_terrain_column, relabel_isaac_backend_manager_output, sdk_joint_indices,
    validate_action_term_contract,
)
from train.common.task import compute_reward_tensor, local_base_clearance

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


# Partitioned source environment.
class Go2IsaacEnv(Go2Environment):
    def __init__(self, config, backend=None):
        validate_environment_contract(config.environment)
        self._owns_backend = backend is None
        if backend is None:
            from isaaclab.envs import ManagerBasedRLEnv

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
        super().__init__(config, config.environment.nr_envs)
        self._select_playback_terrain()
        self._domain_randomization = bool(self.config.domain_randomization)
        self.nr_task_envs = int(self.config.nr_task_envs)
        self.nr_safety_envs = int(self.config.nr_safety_envs)
        if self.nr_task_envs + self.nr_safety_envs != self.nr_envs:
            raise ValueError("Isaac task and safety pool sizes must sum to nr_envs")
        robot = self.backend.scene["robot"]
        validate_action_term_contract(
            self.backend.action_manager.get_term("joint_pos")
        )
        self._joint_indices = sdk_joint_indices(robot.joint_names, robot.device)
        foot_indices, foot_names = robot.find_bodies(".*_foot")
        if len(foot_indices) != 4:
            raise RuntimeError(
                f"Expected four Go2 foot bodies, got {foot_names}"
            )
        self._foot_body_indices = torch.as_tensor(
            foot_indices, dtype=torch.long, device=robot.device
        )
        self._previous_target = default_joint_target(self.nr_envs, robot.device)
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
        print(
            format_policy_io_contract(self.config.target_velocity_x),
            flush=True,
        )

    def _select_playback_terrain(self):
        """Place all playback envs on the requested generated terrain family."""


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
        self._previous_quaternion = None
        self._velocity_estimator.reset()
        self._fall_detector.reset()
        self._episode_return.zero_()
        self._episode_length.zero_()
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
        foot_position = robot.data.body_pos_w[:, self._foot_body_indices]
        foot_hit_distance = (
            foot_position[..., None, :2] - terrain_hits[:, None, :, :2]
        ).square().sum(dim=-1)
        foot_hit_indices = foot_hit_distance.argmin(dim=-1)
        foot_ground_height = torch.gather(
            terrain_hits[..., 2], 1, foot_hit_indices
        )
        foot_clearance = foot_position[..., 2] - foot_ground_height

        imu = self.backend.scene["imu"]
        torque = robot.data.applied_torque[:, self._joint_indices]
        reward_terms, reward = compute_reward_tensor(
            body_velocity,
            imu.data.quat_w,
            imu.data.ang_vel_b,
            torque,
            target_velocity,
        )
        failure = self._fall_detector.update(
            self.backend.scene["imu"].data.quat_w,
            base_clearance,
        )
        failure = failure.clone()
        tilt_failure = self._fall_detector.last_tilt_failure.clone()
        height_failure = self._fall_detector.last_height_failure.clone()
        terminated = backend_terminated | failure
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
            "action_saturation_ratio": (
                applied_action.abs() > 0.98
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
            "reward/total": reward.detach().cpu().numpy(),
            "final_observation": [None] * self.nr_envs,
            "final_info": [None] * self.nr_envs,
        }
        for index in torch.nonzero(done).flatten().tolist():
            info["final_observation"][index] = observation[index].detach().cpu().numpy()
            info["final_info"][index] = {
                "episode_return": float(self._episode_return[index]),
                "episode_length": int(self._episode_length[index]),
            }
            self._episode_return[index] = 0
            self._episode_length[index] = 0
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


    def close(self):
        if self._owns_backend:
            self.backend.close()
