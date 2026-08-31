"""Shared Go2 kinematics and velocity estimation."""
from __future__ import annotations

from dataclasses import asdict, dataclass

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

