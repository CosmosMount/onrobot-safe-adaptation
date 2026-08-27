"""Sensor-free proprioceptive body-velocity estimator shared by both backends."""

from __future__ import annotations

import numpy as np

from ..reward import quaternion_to_rpy_wxyz
from ..types import RobotState
from .kinematics import supported_body_velocity
from .support import infer_support_mask


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
    def __init__(
        self,
        dt: float = 0.02,
        support_gain: float = 0.35,
        no_support_damping: float = 0.995,
    ):
        self.dt = float(dt)
        self.support_gain = float(support_gain)
        self.no_support_damping = float(no_support_damping)
        self.world_velocity = np.zeros(3, dtype=np.float64)

    def reset(self, world_velocity: np.ndarray | None = None) -> None:
        self.world_velocity = np.zeros(3, dtype=np.float64)
        if world_velocity is not None:
            self.world_velocity[:] = np.asarray(world_velocity, dtype=np.float64)

    def update(self, state: RobotState) -> np.ndarray:
        rotation_world_from_body = quaternion_rotation_matrix_wxyz(state.imu_quat)
        if state.imu_accelerometer is not None:
            specific_force_body = np.asarray(
                state.imu_accelerometer, dtype=np.float64
            )
            acceleration_world = rotation_world_from_body @ specific_force_body
            acceleration_world += np.asarray([0.0, 0.0, -9.81])
            self.world_velocity += acceleration_world * self.dt

        support_mask = infer_support_mask(state.joint_q, state.joint_dq)
        support_estimate_body = supported_body_velocity(
            state.joint_q, state.joint_dq, state.imu_gyro, support_mask
        )
        if support_estimate_body is not None:
            support_estimate_world = rotation_world_from_body @ support_estimate_body
            gain = self.support_gain
            self.world_velocity = (
                (1.0 - gain) * self.world_velocity + gain * support_estimate_world
            )
            self.world_velocity[2] = support_estimate_world[2]
        else:
            self.world_velocity *= self.no_support_damping

        # The estimator exposes the body-frame velocity required by the policy.
        return (rotation_world_from_body.T @ self.world_velocity).astype(np.float32)
