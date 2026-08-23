"""The versioned ``total-track-xy`` reward used by both simulators."""

from __future__ import annotations

import math

import numpy as np

from .types import RewardTerms


REWARD_VERSION = "total-track-xy-v1"


def quaternion_to_rpy_wxyz(quaternion: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(pitch_argument)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def track_x_reward(velocity_x: float, target_velocity_x: float) -> float:
    if target_velocity_x <= 0:
        raise ValueError("target_velocity_x must be positive")
    if target_velocity_x <= velocity_x <= 2.0 * target_velocity_x:
        return 1.0
    if velocity_x <= -target_velocity_x or velocity_x >= 4.0 * target_velocity_x:
        return 0.0
    return 1.0 - abs(velocity_x - target_velocity_x) / (
        2.0 * target_velocity_x
    )


def compute_reward(
    world_velocity: np.ndarray,
    imu_quat: np.ndarray,
    imu_gyro: np.ndarray,
    actuator_torque: np.ndarray,
    target_velocity_x: float,
) -> RewardTerms:
    roll, pitch, yaw_angle = quaternion_to_rpy_wxyz(imu_quat)
    vx_world, vy_world = np.asarray(world_velocity, dtype=np.float64)[:2]
    cos_yaw = math.cos(yaw_angle)
    sin_yaw = math.sin(yaw_angle)
    body_vx = cos_yaw * vx_world + sin_yaw * vy_world
    body_vy = -sin_yaw * vx_world + cos_yaw * vy_world
    track_x = track_x_reward(float(body_vx), target_velocity_x)
    track_xy = track_x - abs(float(body_vy))
    yaw = float(np.asarray(imu_gyro)[2]) ** 2
    upright = roll**2 + pitch**2
    energy = float(np.square(np.asarray(actuator_torque, dtype=np.float64)).sum())
    total = max(track_xy - 0.1 * yaw - 10.0 * upright - 0.0003 * energy, 0.0)
    return RewardTerms(track_x, track_xy, yaw, upright, energy, total)

