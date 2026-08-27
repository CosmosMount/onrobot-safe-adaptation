"""Torch-vectorized robust proprioceptive velocity Kalman filter."""

from __future__ import annotations

import torch

from .kinematics import (
    CALF_LENGTH,
    HIP_ABDUCTION_Y,
    HIP_LATERAL_OFFSET,
    HIP_X,
    LEG_SIDE,
    THIGH_LENGTH,
)
from .velocity import (
    DEFAULT_VELOCITY_ESTIMATOR_CONFIG,
    VelocityEstimatorConfig,
)


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
    leg_side = torch.as_tensor(LEG_SIDE, dtype=q.dtype, device=q.device)
    lateral = leg_side * HIP_LATERAL_OFFSET
    total = thigh + calf
    x = hip_x - THIGH_LENGTH * torch.sin(thigh) - CALF_LENGTH * torch.sin(total)
    z_plane = -THIGH_LENGTH * torch.cos(thigh) - CALF_LENGTH * torch.cos(total)
    y = (
        leg_side * HIP_ABDUCTION_Y
        + lateral * torch.cos(abduction)
        - z_plane * torch.sin(abduction)
    )
    z = lateral * torch.sin(abduction) + z_plane * torch.cos(abduction)
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
        (-lateral * torch.sin(abduction) - z_plane * torch.cos(abduction))
        * abduction_dq
        - dz_plane * torch.sin(abduction)
    )
    dz = (
        (lateral * torch.cos(abduction) - z_plane * torch.sin(abduction))
        * abduction_dq
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
        config: VelocityEstimatorConfig = DEFAULT_VELOCITY_ESTIMATOR_CONFIG,
    ):
        self.nr_envs = int(nr_envs)
        self.device = device
        if not torch.isfinite(torch.as_tensor(dt)) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.dt = float(dt)
        self.config = config
        for name, value in vars(config).items():
            setattr(self, name, float(value))
        self.world_velocity = torch.zeros(self.nr_envs, 3, device=device)
        identity = torch.eye(3, device=device)
        self.covariance = identity.expand(self.nr_envs, -1, -1).clone()
        self.covariance.mul_(self.initial_variance)
        self.last_support_confidence = torch.zeros(
            self.nr_envs, 4, device=device
        )
        self.last_innovation_squared = torch.full(
            (self.nr_envs,), torch.nan, device=device
        )
        self.last_measurement_accepted = torch.zeros(
            self.nr_envs, dtype=torch.bool, device=device
        )
        self.update_count = 0

    def reset(self, env_ids=None):
        if env_ids is None:
            self.world_velocity.zero_()
            self.covariance.copy_(
                torch.eye(
                    3,
                    dtype=self.covariance.dtype,
                    device=self.covariance.device,
                )
                .expand(self.nr_envs, -1, -1)
                * self.initial_variance
            )
            self.last_support_confidence.zero_()
            self.last_innovation_squared.fill_(torch.nan)
            self.last_measurement_accepted.zero_()
            self.update_count = 0
        else:
            self.world_velocity[env_ids] = 0
            self.covariance[env_ids] = (
                torch.eye(
                    3,
                    dtype=self.covariance.dtype,
                    device=self.covariance.device,
                )
                * self.initial_variance
            )
            self.last_support_confidence[env_ids] = 0
            self.last_innovation_squared[env_ids] = torch.nan
            self.last_measurement_accepted[env_ids] = False

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
        identity = torch.eye(3, dtype=joint_q.dtype, device=joint_q.device)
        self.covariance.add_(identity * self.process_variance * self.dt * self.dt)

        positions, joint_foot_velocity = _foot_position_velocity(joint_q, joint_dq)
        rotational_velocity = torch.linalg.cross(
            imu_gyro[:, None, :].expand_as(positions), positions, dim=-1
        )
        candidates = -(joint_foot_velocity + rotational_velocity)

        height_delta = positions[..., 2] - positions[..., 2].amin(
            dim=1, keepdim=True
        )
        height_confidence = torch.exp(
            -0.5 * (height_delta / self.height_scale).square()
        )
        vertical_confidence = torch.exp(
            -0.5
            * (
                joint_foot_velocity[..., 2]
                / self.vertical_velocity_scale
            ).square()
        )
        confidence = height_confidence * vertical_confidence
        self.last_support_confidence.copy_(confidence)

        confidence_sum = confidence.sum(dim=1, keepdim=True)
        predicted_body_velocity = torch.bmm(
            rotation.transpose(1, 2), self.world_velocity.unsqueeze(-1)
        ).squeeze(-1)
        residual_norm = torch.linalg.vector_norm(
            candidates - predicted_body_velocity[:, None, :], dim=-1
        )
        huber_weight = torch.where(
            residual_norm > self.huber_delta,
            self.huber_delta / residual_norm.clamp_min(1e-12),
            torch.ones_like(residual_norm),
        )
        prior_likelihood = torch.exp(
            -(
                residual_norm - residual_norm.amin(dim=1, keepdim=True)
            )
            / self.prior_temperature
        )
        weights = confidence.sqrt().sqrt() * huber_weight * prior_likelihood
        weight_sum = weights.sum(dim=1, keepdim=True)
        observed_body_velocity = (
            candidates * weights.unsqueeze(-1)
        ).sum(dim=1) / weight_sum.clamp_min(1e-12)
        residual = candidates - observed_body_velocity[:, None, :]
        spread = (
            residual.square() * weights.unsqueeze(-1)
        ).sum(dim=1) / weight_sum.clamp_min(1e-12)
        effective_count = weight_sum.squeeze(-1).square() / weights.square().sum(
            dim=1
        ).clamp_min(1e-12)
        measurement_diagonal = (
            self.leg_variance / effective_count.clamp_min(1.0)
        ).unsqueeze(-1) + spread
        measurement_covariance_body = torch.diag_embed(measurement_diagonal)
        measurement_covariance_world = torch.bmm(
            torch.bmm(rotation, measurement_covariance_body),
            rotation.transpose(1, 2),
        )
        observed_world_velocity = torch.bmm(
            rotation, observed_body_velocity.unsqueeze(-1)
        ).squeeze(-1)
        innovation = observed_world_velocity - self.world_velocity
        innovation_covariance = self.covariance + measurement_covariance_world
        solved_innovation = torch.linalg.solve(
            innovation_covariance, innovation.unsqueeze(-1)
        ).squeeze(-1)
        innovation_squared = (innovation * solved_innovation).sum(dim=-1)
        has_support = (
            confidence_sum.squeeze(-1) >= self.minimum_total_confidence
        )
        accepted = (
            has_support
            & torch.isfinite(innovation_squared)
            & (innovation_squared <= self.innovation_gate)
        )
        self.last_innovation_squared.copy_(
            torch.where(
                has_support,
                innovation_squared,
                torch.full_like(innovation_squared, torch.nan),
            )
        )
        self.last_measurement_accepted.copy_(accepted)

        kalman_gain = torch.linalg.solve(
            innovation_covariance.transpose(1, 2),
            self.covariance.transpose(1, 2),
        ).transpose(1, 2)
        updated_velocity = self.world_velocity + torch.bmm(
            kalman_gain, innovation.unsqueeze(-1)
        ).squeeze(-1)
        identity_minus_gain = identity.unsqueeze(0) - kalman_gain
        updated_covariance = torch.bmm(
            torch.bmm(identity_minus_gain, self.covariance),
            identity_minus_gain.transpose(1, 2),
        ) + torch.bmm(
            torch.bmm(kalman_gain, measurement_covariance_world),
            kalman_gain.transpose(1, 2),
        )
        updated_covariance = 0.5 * (
            updated_covariance + updated_covariance.transpose(1, 2)
        )
        inflated_covariance = (
            self.covariance * self.rejection_covariance_inflation
        )
        maximum_diagonal = torch.diagonal(
            inflated_covariance, dim1=-2, dim2=-1
        ).amax(dim=-1)
        inflation_cap = torch.clamp(
            self.initial_variance / maximum_diagonal.clamp_min(1e-12),
            max=1.0,
        )
        inflated_covariance = (
            inflated_covariance * inflation_cap[:, None, None]
        )
        rejected_with_measurement = has_support & torch.isfinite(
            innovation_squared
        ) & ~accepted
        self.world_velocity.copy_(
            torch.where(accepted.unsqueeze(-1), updated_velocity, self.world_velocity)
        )
        self.covariance.copy_(
            torch.where(
                accepted[:, None, None],
                updated_covariance,
                torch.where(
                    rejected_with_measurement[:, None, None],
                    inflated_covariance,
                    self.covariance,
                ),
            )
        )
        self.update_count += 1
        return torch.bmm(
            rotation.transpose(1, 2), self.world_velocity.unsqueeze(-1)
        ).squeeze(-1)
