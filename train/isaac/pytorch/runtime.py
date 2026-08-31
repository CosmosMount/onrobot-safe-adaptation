"""Isaac backend stepping and physics-frame components."""

from dataclasses import dataclass

import torch


class PhysicsFrameReportingManagerMixin:
    """IsaacLab step implementation with a post-physics-frame callback.

    IsaacLab normally exposes state only after the decimated policy step and
    may reset timed-out environments before returning.  The callback runs
    immediately after each ``scene.update(physics_dt)`` and therefore lets the
    Go2 adapter retain all ten terminal physics frames before any manager reset.
    """

    def set_physics_frame_callback(self, callback) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("physics frame callback must be callable or None")
        self._physics_frame_callback = callback

    def reset_envs(self, env_ids) -> None:
        """Public indexed reset preserving IsaacLab recorder semantics."""

        env_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        if env_ids.numel() == 0:
            return
        self.recorder_manager.record_pre_reset(env_ids)
        self._reset_idx(env_ids)
        # Match ManagerBasedEnv.reset(): reset terms only stage state in the
        # scene buffers.  Write it to PhysX and run forward kinematics before
        # any sensor render or post-reset observation is allowed to see it.
        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
        self.recorder_manager.record_post_reset(env_ids)

    def step(self, action):
        """Mirror IsaacLab 2.3's manager step and report every physics frame."""

        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            callback = getattr(self, "_physics_frame_callback", None)
            if callback is not None:
                callback()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_envs(reset_env_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )


@dataclass(frozen=True)
class IsaacPhysicsFrame:
    """Cloned pre-reset state for one 2 ms Isaac physics frame."""

    joint_q: torch.Tensor
    joint_dq: torch.Tensor
    imu_gyro: torch.Tensor
    reward_imu_gyro: torch.Tensor
    imu_acceleration: torch.Tensor
    imu_quat: torch.Tensor
    estimated_body_velocity: torch.Tensor
    body_velocity: torch.Tensor
    torque: torch.Tensor
    base_clearance: torch.Tensor
    terrain_height: torch.Tensor
    foot_clearance: torch.Tensor


class TorchFallDetector:
    """Vectorized common tilt-or-low-base SQRL incident detector."""

    def __init__(
        self,
        nr_envs: int,
        device,
        angle_threshold: float = 0.8,
        min_base_clearance: float = 0.18,
        consecutive_frames: int = 5,
    ):
        if consecutive_frames < 1:
            raise ValueError("consecutive_frames must be at least 1")
        self.angle_threshold = float(angle_threshold)
        self.min_base_clearance = float(min_base_clearance)
        self.consecutive_frames = int(consecutive_frames)
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
            self.tilt_count + 1,
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
            self.height_count + 1,
            torch.zeros_like(self.height_count),
        )
        self.last_tilt_failure = self.tilt_count >= self.consecutive_frames
        self.last_height_failure = self.height_count >= self.consecutive_frames
        return self.last_tilt_failure | self.last_height_failure

