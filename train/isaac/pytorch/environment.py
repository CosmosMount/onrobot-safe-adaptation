"""Isaac Lab implementation of the shared Go2 environment."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import fields
from io import StringIO

import numpy as np
import torch

from train.core.base import (
    ACTION_SIZE, ACTION_SPEC, DEFAULT_JOINT_POSITION, Go2Environment,
    PHYSICS_DT, PHYSICS_STEPS_PER_ACTION, format_policy_io_contract,
    project_action_targets_tensor, validate_environment_contract,
)
from train.core.estimation import TorchVelocityEstimator, velocity_estimator_config_from
from train.core.task import compute_reward_tensor, local_base_clearance

from .contract import (
    build_observation_tensor,
    default_joint_target,
    sdk_joint_indices,
    validate_action_term_contract,
)
from .randomization import (
    TRANSFER_RANDOMIZATION,
    format_domain_randomization_report,
)
from .runtime import (
    IsaacPhysicsFrame,
    PhysicsFrameReportingManagerMixin,
    TorchFallDetector,
)
from .setup import make_env_cfg, relabel_isaac_backend_manager_output
from .terrain import playback_terrain_column


# Child-local Isaac vector environment.
class Go2IsaacEnv(Go2Environment):
    def __init__(self, config, backend=None):
        validate_environment_contract(config.environment)
        self._owns_backend = backend is None
        if backend is None:
            from isaaclab.envs import ManagerBasedRLEnv

            class ReportedManagerBasedRLEnv(
                PhysicsFrameReportingManagerMixin, ManagerBasedRLEnv
            ):
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
        set_frame_callback = getattr(
            self.backend, "set_physics_frame_callback", None
        )
        reset_envs = getattr(self.backend, "reset_envs", None)
        if not callable(set_frame_callback) or not callable(reset_envs):
            raise RuntimeError(
                "Isaac backend must implement set_physics_frame_callback() "
                "and public reset_envs(); a plain ManagerBasedRLEnv cannot "
                "preserve terminal physics frames safely"
            )
        self._previous_target = default_joint_target(self.nr_envs, robot.device)
        self._previous_quaternion = None
        self._velocity_estimator = TorchVelocityEstimator(
            self.nr_envs,
            robot.device,
            dt=PHYSICS_DT,
            config=velocity_estimator_config_from(self.config),
        )
        self._fall_detector = TorchFallDetector(
            self.nr_envs,
            robot.device,
            angle_threshold=float(self.config.fall_angle_threshold),
            min_base_clearance=float(self.config.fall_min_base_clearance),
            consecutive_frames=int(self.config.fall_consecutive_frames),
        )
        self._physics_frames: list[IsaacPhysicsFrame] = []
        self._collect_physics_frames = False
        self._window_failure = torch.zeros(
            self.nr_envs, dtype=torch.bool, device=robot.device
        )
        self._window_tilt_failure = self._window_failure.clone()
        self._window_height_failure = self._window_failure.clone()
        self._first_failure_frame = torch.full(
            (self.nr_envs,), -1, dtype=torch.long, device=robot.device
        )
        self._latest_estimated_body_velocity = torch.zeros(
            self.nr_envs, 3, device=robot.device
        )
        set_frame_callback(self._on_physics_frame)
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

    def _terrain_geometry(self):
        robot = self._robot
        terrain_hits = torch.nan_to_num(
            self.backend.scene["height_scanner"].data.ray_hits_w,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if terrain_hits.ndim != 3 or terrain_hits.shape[1] < 1:
            raise RuntimeError(
                "Isaac height scanner must provide at least one terrain ray per env"
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
        return base_clearance, terrain_height, foot_clearance

    def _capture_current_frame(self) -> IsaacPhysicsFrame:
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
        estimated_body_velocity = self._velocity_estimator.update(
            joint_q,
            joint_dq,
            imu_gyro,
            imu.data.quat_w,
            imu_acceleration,
        )
        base_clearance, terrain_height, foot_clearance = self._terrain_geometry()
        return IsaacPhysicsFrame(
            joint_q=joint_q.clone(),
            joint_dq=joint_dq.clone(),
            imu_gyro=imu_gyro.clone(),
            reward_imu_gyro=imu.data.ang_vel_b.clone(),
            imu_acceleration=imu_acceleration.clone(),
            imu_quat=imu.data.quat_w.clone(),
            estimated_body_velocity=estimated_body_velocity.clone(),
            body_velocity=robot.data.root_lin_vel_b.clone(),
            torque=robot.data.applied_torque[:, self._joint_indices].clone(),
            base_clearance=base_clearance.clone(),
            terrain_height=terrain_height.clone(),
            foot_clearance=foot_clearance.clone(),
        )

    def _on_physics_frame(self) -> None:
        if not self._collect_physics_frames:
            return
        frame = self._capture_current_frame()
        failure = self._fall_detector.update(
            frame.imu_quat, frame.base_clearance
        )
        first_failure = failure & ~self._window_failure
        self._first_failure_frame[first_failure] = len(self._physics_frames)
        self._window_tilt_failure[first_failure] = (
            self._fall_detector.last_tilt_failure[first_failure]
        )
        self._window_height_failure[first_failure] = (
            self._fall_detector.last_height_failure[first_failure]
        )
        self._window_failure |= failure
        self._physics_frames.append(frame)

    def _transition_frame(self) -> IsaacPhysicsFrame:
        """Select each lane's first failure frame, or its final normal frame."""

        if not bool(self._window_failure.any()):
            return self._physics_frames[-1]
        final_index = len(self._physics_frames) - 1
        frame_indices = torch.where(
            self._first_failure_frame >= 0,
            self._first_failure_frame,
            torch.full_like(self._first_failure_frame, final_index),
        )
        env_indices = torch.arange(self.nr_envs, device=self._robot.device)
        values = {
            field.name: torch.stack(
                [getattr(frame, field.name) for frame in self._physics_frames]
            )[frame_indices, env_indices]
            for field in fields(IsaacPhysicsFrame)
        }
        return IsaacPhysicsFrame(**values)

    def _observation_from_frame(self, frame: IsaacPhysicsFrame):
        observation, quaternion = build_observation_tensor(
            frame.joint_q,
            frame.joint_dq,
            frame.imu_gyro,
            frame.estimated_body_velocity,
            frame.imu_quat,
            self._previous_target,
            self._previous_quaternion,
        )
        self._previous_quaternion = quaternion
        return observation

    def _normalize_env_ids(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(
                self.nr_envs, device=self._robot.device
            )[env_ids]
        else:
            env_ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self._robot.device
            ).reshape(-1)
        if env_ids.numel() == 0:
            return env_ids
        if bool(((env_ids < 0) | (env_ids >= self.nr_envs)).any()):
            raise IndexError(f"Isaac reset env ids out of range: {env_ids.tolist()}")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("Isaac reset env ids must be unique")
        return env_ids

    def _reset_adapter_state(self, env_ids, *, reset_accounting=True):
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self._previous_target[env_ids] = torch.as_tensor(
            DEFAULT_JOINT_POSITION,
            dtype=self._previous_target.dtype,
            device=self._previous_target.device,
        )
        self._velocity_estimator.reset(env_ids)
        self._fall_detector.reset(env_ids)
        self._latest_estimated_body_velocity[env_ids] = 0
        current_all = self.backend.scene["imu"].data.quat_w.clone()
        current_all = current_all / torch.linalg.vector_norm(
            current_all, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        current_all = torch.where(current_all[:, :1] < 0, -current_all, current_all)
        if self._previous_quaternion is None:
            self._previous_quaternion = current_all
        else:
            self._previous_quaternion[env_ids] = current_all[env_ids]
        if reset_accounting:
            self._episode_return[env_ids] = 0
            self._episode_length[env_ids] = 0

    def _current_reset_observation(self, env_ids):
        env_ids = self._normalize_env_ids(env_ids)
        robot = self._robot
        imu = self.backend.scene["imu"]
        joint_q = self._sensor_value(
            robot.data.joint_pos[env_ids][:, self._joint_indices],
            "joint_position_noise",
        )
        joint_dq = self._sensor_value(
            robot.data.joint_vel[env_ids][:, self._joint_indices],
            "joint_velocity_noise",
        )
        imu_gyro = self._sensor_value(
            imu.data.ang_vel_b[env_ids], "gyro_noise"
        )
        observation, quaternion = build_observation_tensor(
            joint_q,
            joint_dq,
            imu_gyro,
            torch.zeros(
                env_ids.numel(), 3, dtype=joint_q.dtype, device=joint_q.device
            ),
            imu.data.quat_w[env_ids],
            self._previous_target[env_ids],
            self._previous_quaternion[env_ids],
        )
        self._previous_quaternion[env_ids] = quaternion
        return observation

    def _validate_reset_contract(self, env_ids) -> None:
        if self._domain_randomization or str(self.config.terrain_mode) != "flat":
            return
        env_ids = self._normalize_env_ids(env_ids)
        robot = self._robot
        joint_q = robot.data.joint_pos[env_ids][:, self._joint_indices]
        expected = torch.as_tensor(
            DEFAULT_JOINT_POSITION, dtype=joint_q.dtype, device=joint_q.device
        )
        joint_error = (joint_q - expected).abs().amax()
        joint_tolerance = float(self.config.reset_joint_validation_tolerance)
        if float(joint_error) > joint_tolerance:
            raise RuntimeError(
                "Isaac reset joint pose violates the shared home contract: "
                f"max error {float(joint_error):.6f} rad > {joint_tolerance:.6f}"
            )
        base_height = robot.data.root_pos_w[env_ids, 2]
        expected_height = float(self.config.reset_base_height)
        height_error = (base_height - expected_height).abs().amax()
        height_tolerance = float(
            self.config.reset_base_height_validation_tolerance
        )
        if float(height_error) > height_tolerance:
            raise RuntimeError(
                "Isaac reset base height violates the shared contract: "
                f"max error {float(height_error):.6f} m > {height_tolerance:.6f}"
            )
        quaternion = self.backend.scene["imu"].data.quat_w[env_ids]
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        orientation_error = 2.0 * torch.acos(
            quaternion[:, 0].abs().clamp(max=1.0)
        ).amax()
        orientation_tolerance = float(
            self.config.reset_orientation_validation_tolerance
        )
        if float(orientation_error) > orientation_tolerance:
            raise RuntimeError(
                "Isaac reset base orientation violates the shared identity "
                f"contract: max error {float(orientation_error):.6f} rad > "
                f"{orientation_tolerance:.6f}"
            )
        _, _, foot_clearance = self._terrain_geometry()
        foot_surface = foot_clearance[env_ids] - float(
            self.config.foot_collision_radius
        )
        foot_tolerance = float(
            self.config.reset_foot_surface_validation_tolerance
        )
        if float(foot_surface.abs().amax()) > foot_tolerance:
            raise RuntimeError(
                "Isaac reset feet are not on the ground within the shared "
                f"surface tolerance: {foot_surface.detach().cpu().tolist()}"
            )

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
        self._latest_estimated_body_velocity.zero_()
        env_ids = torch.arange(self.nr_envs, device=self._robot.device)
        self._reset_adapter_state(env_ids)
        self._validate_reset_contract(env_ids)
        observation = self._current_reset_observation(env_ids)
        return observation.detach().cpu().numpy(), {}

    def step(self, actions):
        action = torch.as_tensor(
            actions, dtype=torch.float32, device=self._robot.device
        ).reshape(self.nr_envs, ACTION_SIZE)
        applied_action, target = project_action_targets_tensor(
            self._previous_target, action
        )
        self._previous_target = target

        self._physics_frames.clear()
        self._window_failure.zero_()
        self._window_tilt_failure.zero_()
        self._window_height_failure.zero_()
        self._first_failure_frame.fill_(-1)
        self._collect_physics_frames = True
        try:
            _, _, backend_terminated, backend_truncated, extras = self.backend.step(
                applied_action
            )
        finally:
            self._collect_physics_frames = False
        backend_terminated = backend_terminated.clone()
        backend_truncated = backend_truncated.clone()
        del extras
        if len(self._physics_frames) != PHYSICS_STEPS_PER_ACTION:
            raise RuntimeError(
                "Isaac backend reported "
                f"{len(self._physics_frames)} physics frames; expected exactly "
                f"{PHYSICS_STEPS_PER_ACTION}"
            )
        terminal_frame = self._transition_frame()
        terminal_observation = self._observation_from_frame(terminal_frame)
        target_velocity = float(self.config.target_velocity_x)
        reward_terms, reward = compute_reward_tensor(
            terminal_frame.body_velocity,
            terminal_frame.imu_quat,
            terminal_frame.reward_imu_gyro,
            terminal_frame.torque,
            target_velocity,
        )
        failure = self._window_failure.clone()
        tilt_failure = self._window_tilt_failure.clone()
        height_failure = self._window_height_failure.clone()
        terminated = backend_terminated | failure
        truncated = backend_truncated & ~terminated
        done = terminated | truncated
        self._episode_return += reward
        self._episode_length += 1
        info = {
            "failure": failure.float().cpu().numpy(),
            "failure/tilt": tilt_failure.float().cpu().numpy(),
            "failure/height": height_failure.float().cpu().numpy(),
            "applied_action": applied_action.detach().cpu().numpy(),
            "forward_velocity": terminal_frame.body_velocity[:, 0].detach().cpu().numpy(),
            "estimated_forward_velocity": terminal_frame.estimated_body_velocity[
                :, 0
            ].detach().cpu().numpy(),
            "target_velocity_error": (
                terminal_frame.body_velocity[:, 0] - target_velocity
            ).abs().detach().cpu().numpy(),
            "base_clearance": terminal_frame.base_clearance.detach().cpu().numpy(),
            "local_terrain_height": terminal_frame.terrain_height.detach().cpu().numpy(),
            "mean_foot_clearance": terminal_frame.foot_clearance.mean(
                dim=-1
            ).detach().cpu().numpy(),
            "max_foot_clearance": terminal_frame.foot_clearance.max(
                dim=-1
            ).values.detach().cpu().numpy(),
            "action_saturation_ratio": (
                applied_action.abs() > 0.98
            ).float().mean(dim=-1).detach().cpu().numpy(),
            "torque_saturation_ratio": (
                terminal_frame.torque.abs()
                > 0.95 * ACTION_SPEC.effort_limit
            ).float().mean(dim=-1).detach().cpu().numpy(),
            "velocity_estimation_error": torch.linalg.vector_norm(
                terminal_frame.estimated_body_velocity
                - terminal_frame.body_velocity,
                dim=-1,
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
            info["final_observation"][index] = (
                terminal_observation[index].detach().cpu().numpy()
            )
            info["final_info"][index] = {
                "episode_return": float(self._episode_return[index]),
                "episode_length": int(self._episode_length[index]),
            }
        already_reset = backend_terminated | backend_truncated
        reset_ids = torch.nonzero(done & ~already_reset).flatten()
        if reset_ids.numel() > 0:
            self.backend.reset_envs(reset_ids)
        done_ids = torch.nonzero(done).flatten()
        observation = terminal_observation.clone()
        if done_ids.numel() > 0:
            self._reset_adapter_state(done_ids)
            self._validate_reset_contract(done_ids)
            observation[done_ids] = self._current_reset_observation(done_ids)
        return (
            observation.detach().cpu().numpy(),
            reward.detach().cpu().numpy(),
            terminated.detach().cpu().numpy(),
            truncated.detach().cpu().numpy(),
            info,
        )

    def close(self):
        if self._owns_backend:
            self.backend.close()
