"""Torch-vectorized sensor-free proprioceptive velocity estimator."""

from __future__ import annotations

import torch

from .kinematics import CALF_LENGTH, HIP_X, HIP_Y, THIGH_LENGTH


def quaternion_rotation_matrix_wxyz_torch(quaternion):
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(dim=-1)
    row0 = torch.stack(
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        dim=-1,
    )
    row1 = torch.stack(
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        dim=-1,
    )
    row2 = torch.stack(
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        dim=-1,
    )
    return torch.stack((row0, row1, row2), dim=-2)


def _foot_position_velocity(joint_q, joint_dq):
    q = joint_q.reshape(-1, 4, 3)
    dq = joint_dq.reshape(-1, 4, 3)
    abduction, thigh, calf = q.unbind(dim=-1)
    thigh_dq, calf_dq = dq[..., 1], dq[..., 2]
    hip_x = torch.as_tensor(HIP_X, dtype=q.dtype, device=q.device)
    hip_y = torch.as_tensor(HIP_Y, dtype=q.dtype, device=q.device)
    total = thigh + calf
    x = hip_x - THIGH_LENGTH * torch.sin(thigh) - CALF_LENGTH * torch.sin(total)
    z_plane = -THIGH_LENGTH * torch.cos(thigh) - CALF_LENGTH * torch.cos(total)
    y = hip_y - z_plane * torch.sin(abduction)
    z = z_plane * torch.cos(abduction)
    position = torch.stack((x, y, z), dim=-1)

    dx = (
        (-THIGH_LENGTH * torch.cos(thigh) - CALF_LENGTH * torch.cos(total))
        * thigh_dq
        - CALF_LENGTH * torch.cos(total) * calf_dq
    )
    dz_plane = (
        (THIGH_LENGTH * torch.sin(thigh) + CALF_LENGTH * torch.sin(total))
        * thigh_dq
        + CALF_LENGTH * torch.sin(total) * calf_dq
    )
    abduction_dq = dq[..., 0]
    dy = (
        -z_plane * torch.cos(abduction) * abduction_dq
        - dz_plane * torch.sin(abduction)
    )
    dz = (
        -z_plane * torch.sin(abduction) * abduction_dq
        + dz_plane * torch.cos(abduction)
    )
    velocity = torch.stack((dx, dy, dz), dim=-1)
    return position, velocity


class TorchVelocityEstimator:
    def __init__(
        self,
        nr_envs: int,
        device,
        dt: float = 0.02,
        support_gain: float = 0.35,
        no_support_damping: float = 0.995,
    ):
        self.dt = float(dt)
        self.support_gain = float(support_gain)
        self.no_support_damping = float(no_support_damping)
        self.world_velocity = torch.zeros(nr_envs, 3, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.world_velocity.zero_()
        else:
            self.world_velocity[env_ids] = 0

    @torch.no_grad()
    def update(
        self,
        joint_q,
        joint_dq,
        imu_gyro,
        imu_quat,
        imu_accelerometer,
    ):
        rotation = quaternion_rotation_matrix_wxyz_torch(imu_quat)
        acceleration_world = torch.bmm(
            rotation, imu_accelerometer.unsqueeze(-1)
        ).squeeze(-1)
        acceleration_world = acceleration_world + torch.tensor(
            [0.0, 0.0, -9.81], device=joint_q.device, dtype=joint_q.dtype
        )
        self.world_velocity.add_(acceleration_world * self.dt)

        positions, joint_foot_velocity = _foot_position_velocity(joint_q, joint_dq)
        rotational_velocity = torch.linalg.cross(
            imu_gyro[:, None, :].expand_as(positions), positions, dim=-1
        )
        supported_velocity_body = -(joint_foot_velocity + rotational_velocity)
        lowest = positions[..., 2] <= positions[..., 2].amin(
            dim=1, keepdim=True
        ) + 0.035
        slow_vertical = joint_foot_velocity[..., 2].abs() < 0.35
        support_mask = lowest & slow_vertical
        weights = support_mask.to(joint_q.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1)
        has_support = counts[:, 0] > 0
        support_body = (supported_velocity_body * weights).sum(dim=1) / counts.clamp_min(1)
        support_world = torch.bmm(rotation, support_body.unsqueeze(-1)).squeeze(-1)
        gain = self.support_gain
        self.world_velocity[has_support] = (
            (1.0 - gain) * self.world_velocity[has_support]
            + gain * support_world[has_support]
        )
        self.world_velocity[has_support, 2] = support_world[has_support, 2]
        self.world_velocity[~has_support] *= self.no_support_damping
        return torch.bmm(
            rotation.transpose(1, 2), self.world_velocity.unsqueeze(-1)
        ).squeeze(-1)
