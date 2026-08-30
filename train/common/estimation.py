"""Shared Go2 kinematics and velocity estimation."""
from __future__ import annotations

import numpy as np

from .base import RobotState

# Shared kinematics and proprioceptive velocity estimation.
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
HIP_X = np.asarray([0.1934, 0.1934, -0.1934, -0.1934], dtype=np.float64)
LEG_SIDE = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
HIP_ABDUCTION_Y = 0.0465
HIP_LATERAL_OFFSET = 0.0955


def foot_position_velocity_body(
    joint_q: np.ndarray, joint_dq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact Go2 foot positions and joint-induced velocities."""

    q = np.asarray(joint_q, dtype=np.float64).reshape(4, 3)
    dq = np.asarray(joint_dq, dtype=np.float64).reshape(4, 3)
    abduction, thigh, calf = q.T
    abduction_dq, thigh_dq, calf_dq = dq.T
    total = thigh + calf
    lateral = LEG_SIDE * HIP_LATERAL_OFFSET
    x_plane = -THIGH_LENGTH * np.sin(thigh) - CALF_LENGTH * np.sin(total)
    z_plane = -THIGH_LENGTH * np.cos(thigh) - CALF_LENGTH * np.cos(total)
    positions = np.stack(
        (
            HIP_X + x_plane,
            LEG_SIDE * HIP_ABDUCTION_Y
            + lateral * np.cos(abduction)
            - z_plane * np.sin(abduction),
            lateral * np.sin(abduction) + z_plane * np.cos(abduction),
        ),
        axis=-1,
    )

    dx = (
        (-THIGH_LENGTH * np.cos(thigh) - CALF_LENGTH * np.cos(total))
        * thigh_dq
        - CALF_LENGTH * np.cos(total) * calf_dq
    )
    dz_plane = (
        (THIGH_LENGTH * np.sin(thigh) + CALF_LENGTH * np.sin(total))
        * thigh_dq
        + CALF_LENGTH * np.sin(total) * calf_dq
    )
    velocities = np.stack(
        (
            dx,
            (-lateral * np.sin(abduction) - z_plane * np.cos(abduction))
            * abduction_dq
            - dz_plane * np.sin(abduction),
            (lateral * np.cos(abduction) - z_plane * np.sin(abduction))
            * abduction_dq
            + dz_plane * np.cos(abduction),
        ),
        axis=-1,
    )
    return positions, velocities

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VelocityEstimatorConfig:
    """Backend-independent contact-free Kalman-filter parameters."""

    process_variance: float = 0.03059
    leg_variance: float = 0.002
    initial_variance: float = 0.1
    height_scale: float = 0.05
    vertical_velocity_scale: float = 0.35
    huber_delta: float = 0.25
    prior_temperature: float = 0.05
    innovation_gate: float = 11.34
    rejection_covariance_inflation: float = 2.0
    minimum_total_confidence: float = 0.2

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.rejection_covariance_inflation <= 1.0:
            raise ValueError("rejection_covariance_inflation must be greater than one")


DEFAULT_VELOCITY_ESTIMATOR_CONFIG = VelocityEstimatorConfig()
_CONFIG_PREFIX = "velocity_estimator_"


def configure_velocity_estimator(config) -> None:
    """Expose the common estimator settings on an environment ConfigDict."""

    for name, value in asdict(DEFAULT_VELOCITY_ESTIMATOR_CONFIG).items():
        setattr(config, f"{_CONFIG_PREFIX}{name}", value)


def velocity_estimator_config_from(config) -> VelocityEstimatorConfig:
    """Build immutable estimator settings from an environment ConfigDict."""

    return VelocityEstimatorConfig(
        **{
            name: float(getattr(config, f"{_CONFIG_PREFIX}{name}"))
            for name in asdict(DEFAULT_VELOCITY_ESTIMATOR_CONFIG)
        }
    )


def quaternion_rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("IMU quaternion is not finite and normalized")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class VelocityEstimator:
    """Three-dimensional IMU/leg-odometry KF without contact sensors."""

    def __init__(
        self,
        *,
        dt: float = 0.02,
        config: VelocityEstimatorConfig = DEFAULT_VELOCITY_ESTIMATOR_CONFIG,
    ):
        if not np.isfinite(dt) or float(dt) <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.dt = float(dt)
        self.config = config
        for name, value in asdict(config).items():
            setattr(self, name, float(value))
        self.world_velocity = np.zeros(3, dtype=np.float64)
        self.covariance = np.eye(3, dtype=np.float64) * self.initial_variance
        self.last_support_confidence = np.zeros(4, dtype=np.float64)
        self.last_innovation_squared: float | None = None
        self.last_measurement_accepted = False
        self.update_count = 0

    def reset(self, world_velocity: np.ndarray | None = None) -> None:
        self.world_velocity.fill(0.0)
        if world_velocity is not None:
            self.world_velocity[:] = self._vector(
                world_velocity, 3, "world_velocity"
            )
        self.covariance = np.eye(3, dtype=np.float64) * self.initial_variance
        self.last_support_confidence.fill(0.0)
        self.last_innovation_squared = None
        self.last_measurement_accepted = False
        self.update_count = 0

    @staticmethod
    def _vector(value, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.size != size:
            raise ValueError(f"{name} must contain {size} values, got {array.shape}")
        array = array.reshape(size)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinity")
        return array

    def _leg_observation(
        self,
        joint_q: np.ndarray,
        joint_dq: np.ndarray,
        angular_velocity_body: np.ndarray,
        predicted_body_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        positions, relative_velocity = foot_position_velocity_body(
            joint_q, joint_dq
        )
        candidates = -(
            relative_velocity
            + np.cross(angular_velocity_body[None, :], positions)
        )

        height_delta = positions[:, 2] - np.min(positions[:, 2])
        confidence = np.exp(
            -0.5 * np.square(height_delta / self.height_scale)
            -0.5
            * np.square(relative_velocity[:, 2] / self.vertical_velocity_scale)
        )
        self.last_support_confidence[:] = confidence
        if float(np.sum(confidence)) < self.minimum_total_confidence:
            return None

        residual_norm = np.linalg.norm(
            candidates - predicted_body_velocity, axis=1
        )
        huber_weight = np.ones(4, dtype=np.float64)
        outside = residual_norm > self.huber_delta
        huber_weight[outside] = self.huber_delta / residual_norm[outside]
        prior_likelihood = np.exp(
            -(residual_norm - np.min(residual_norm)) / self.prior_temperature
        )
        weights = np.sqrt(np.sqrt(confidence)) * huber_weight * prior_likelihood
        weight_sum = float(np.sum(weights))
        if weight_sum <= np.finfo(np.float64).eps:
            return None

        observed_velocity = np.average(candidates, axis=0, weights=weights)
        residual = candidates - observed_velocity
        spread = np.average(np.square(residual), axis=0, weights=weights)
        effective_count = weight_sum * weight_sum / max(
            float(np.dot(weights, weights)), np.finfo(np.float64).eps
        )
        measurement_diagonal = (
            self.leg_variance / max(effective_count, 1.0) + spread
        )
        return observed_velocity, np.diag(measurement_diagonal)

    def update(self, state: RobotState) -> np.ndarray:
        joint_q = self._vector(state.joint_q, 12, "joint_q")
        joint_dq = self._vector(state.joint_dq, 12, "joint_dq")
        angular_velocity = self._vector(state.imu_gyro, 3, "imu_gyro")
        quaternion = self._vector(state.imu_quat, 4, "imu_quat")
        rotation = quaternion_rotation_matrix_wxyz(quaternion)

        if state.imu_accelerometer is not None:
            specific_force = self._vector(
                state.imu_accelerometer, 3, "imu_accelerometer"
            )
            acceleration_world = rotation @ specific_force
            acceleration_world += np.asarray([0.0, 0.0, -9.81])
            self.world_velocity += acceleration_world * self.dt
        self.covariance += (
            np.eye(3, dtype=np.float64)
            * self.process_variance
            * self.dt
            * self.dt
        )
        self.last_innovation_squared = None
        self.last_measurement_accepted = False

        observation = self._leg_observation(
            joint_q,
            joint_dq,
            angular_velocity,
            rotation.T @ self.world_velocity,
        )
        if observation is not None:
            observed_body_velocity, covariance_body = observation
            observed_world_velocity = rotation @ observed_body_velocity
            measurement_covariance = rotation @ covariance_body @ rotation.T
            innovation = observed_world_velocity - self.world_velocity
            innovation_covariance = self.covariance + measurement_covariance
            innovation_squared = float(
                innovation @ np.linalg.solve(innovation_covariance, innovation)
            )
            self.last_innovation_squared = innovation_squared
            if np.isfinite(innovation_squared) and (
                innovation_squared <= self.innovation_gate
            ):
                kalman_gain = np.linalg.solve(
                    innovation_covariance.T, self.covariance.T
                ).T
                self.world_velocity += kalman_gain @ innovation
                identity_minus_gain = np.eye(3) - kalman_gain
                # Joseph form preserves positive semi-definiteness.
                self.covariance = (
                    identity_minus_gain
                    @ self.covariance
                    @ identity_minus_gain.T
                    + kalman_gain @ measurement_covariance @ kalman_gain.T
                )
                self.covariance = 0.5 * (
                    self.covariance + self.covariance.T
                )
                self.last_measurement_accepted = True
            elif np.isfinite(innovation_squared):
                # Repeated coherent innovations must eventually escape a stale
                # covariance gate; one-frame slips still change no velocity.
                inflated = self.covariance * self.rejection_covariance_inflation
                maximum_diagonal = float(np.max(np.diag(inflated)))
                if maximum_diagonal > self.initial_variance:
                    inflated *= self.initial_variance / maximum_diagonal
                self.covariance = 0.5 * (inflated + inflated.T)

        if not np.all(np.isfinite(self.world_velocity)) or not np.all(
            np.isfinite(self.covariance)
        ):
            raise ValueError("Velocity estimator produced NaN or infinity")
        self.update_count += 1
        return (rotation.T @ self.world_velocity).astype(np.float32)


__all__ = [
    "DEFAULT_VELOCITY_ESTIMATOR_CONFIG",
    "VelocityEstimator",
    "VelocityEstimatorConfig",
    "configure_velocity_estimator",
    "quaternion_rotation_matrix_wxyz",
    "velocity_estimator_config_from",
]

import torch


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
        identity = torch.eye(
            3, dtype=self.covariance.dtype, device=self.covariance.device
        )
        if env_ids is None:
            self.world_velocity.zero_()
            self.covariance.copy_(
                identity.expand(self.nr_envs, -1, -1) * self.initial_variance
            )
            self.last_support_confidence.zero_()
            self.last_innovation_squared.fill_(torch.nan)
            self.last_measurement_accepted.zero_()
            self.update_count = 0
        else:
            self.world_velocity[env_ids] = 0
            self.covariance[env_ids] = identity * self.initial_variance
            self.last_support_confidence[env_ids] = 0
            self.last_innovation_squared[env_ids] = torch.nan
            self.last_measurement_accepted[env_ids] = False

    @staticmethod
    def _assert_finite(value, message):
        condition = torch.isfinite(value).all()
        if value.device.type == "cpu":
            if not bool(condition):
                raise ValueError(message)
        else:
            torch._assert_async(condition, message)

    def _validate_inputs(
        self, joint_q, joint_dq, imu_gyro, imu_quat, imu_accelerometer
    ):
        expected = (
            (12, joint_q),
            (12, joint_dq),
            (3, imu_gyro),
            (4, imu_quat),
            (3, imu_accelerometer),
        )
        for width, value in expected:
            if value.ndim != 2 or value.shape != (self.nr_envs, width):
                raise ValueError(
                    "Estimator input must have shape "
                    f"[{self.nr_envs}, {width}], got {tuple(value.shape)}"
                )
            self._assert_finite(
                value, "Velocity estimator input contains NaN or infinity"
            )
        quaternion_valid = (
            torch.linalg.vector_norm(imu_quat, dim=-1) >= 1e-8
        ).all()
        if imu_quat.device.type == "cpu":
            if not bool(quaternion_valid):
                raise ValueError("IMU quaternion is not finite and normalized")
        else:
            torch._assert_async(
                quaternion_valid, "IMU quaternion is not finite and normalized"
            )

    @torch.no_grad()
    def update(
        self,
        joint_q,
        joint_dq,
        imu_gyro,
        imu_quat,
        imu_accelerometer,
    ):
        self._validate_inputs(
            joint_q, joint_dq, imu_gyro, imu_quat, imu_accelerometer
        )
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
        confidence = torch.exp(
            -0.5 * (height_delta / self.height_scale).square()
            -0.5
            * (joint_foot_velocity[..., 2] / self.vertical_velocity_scale).square()
        )
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
        self._assert_finite(
            self.world_velocity, "Velocity estimator produced NaN or infinity"
        )
        self._assert_finite(
            self.covariance, "Velocity estimator produced NaN or infinity"
        )
        self.update_count += 1
        return torch.bmm(
            rotation.transpose(1, 2), self.world_velocity.unsqueeze(-1)
        ).squeeze(-1)
