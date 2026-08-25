"""Command-following FlashSAC Go2 reward shared by both simulators."""

from __future__ import annotations

import math

import numpy as np

from .types import RewardTerms


REWARD_VERSION = "flashsac-go2-walk-easy-command-v3"
REWARD_DT = 0.02
TRACKING_SIGMA = 0.25
BASE_HEIGHT_TARGET = 0.3
REWARD_SCALES = {
    "tracking_lin_vel": 1.0,
    # The source exponential still pays 36.8% of its maximum at zero speed
    # for a 0.5 m/s command.  This explicit error cost makes standing still a
    # negative-return local optimum while leaving exact tracking unchanged.
    "velocity_error": -3.0,
    "tracking_ang_vel": 0.2,
    "lin_vel_z": -1.0,
    "base_height": -50.0,
    "action_rate": -0.005,
    "similar_to_default": -0.1,
}
REWARD_DEFAULT_JOINT_POSITION = np.asarray(
    [
        0.0, 0.8, -1.5,
        0.0, 0.8, -1.5,
        0.0, 1.0, -1.5,
        0.0, 1.0, -1.5,
    ],
    dtype=np.float32,
)


def quaternion_to_rpy_wxyz(
    quaternion: np.ndarray,
) -> tuple[float, float, float]:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def compute_reward(
    world_velocity: np.ndarray,
    imu_quat: np.ndarray,
    imu_gyro: np.ndarray,
    joint_q: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    target_velocity_x: float,
    *,
    base_height: float,
    target_velocity_y: float = 0.0,
    target_angular_velocity_z: float = 0.0,
) -> RewardTerms:
    """Compute the source reward plus a continuous command-error penalty."""

    quaternion = np.asarray(imu_quat, dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-8)
    w, x, y, z = quaternion
    rotation_body_to_world = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    body_velocity = rotation_body_to_world.T @ np.asarray(
        world_velocity, dtype=np.float64
    )[:3]
    command_xy = np.asarray(
        [target_velocity_x, target_velocity_y], dtype=np.float64
    )
    linear_error = float(np.square(command_xy - body_velocity[:2]).sum())
    angular_error = (
        float(target_angular_velocity_z) - float(np.asarray(imu_gyro)[2])
    ) ** 2
    tracking_lin_vel = math.exp(-linear_error / TRACKING_SIGMA)
    tracking_ang_vel = math.exp(-angular_error / TRACKING_SIGMA)
    lin_vel_z = float(body_velocity[2] ** 2)
    base_height_error = (float(base_height) - BASE_HEIGHT_TARGET) ** 2
    action_rate = float(
        np.square(
            np.asarray(previous_action, dtype=np.float64)
            - np.asarray(action, dtype=np.float64)
        ).sum()
    )
    similar_to_default = float(
        np.abs(
            np.asarray(joint_q, dtype=np.float64)
            - np.asarray(REWARD_DEFAULT_JOINT_POSITION, dtype=np.float64)
        ).sum()
    )
    weighted = {
        "tracking_lin_vel": REWARD_DT
        * REWARD_SCALES["tracking_lin_vel"]
        * tracking_lin_vel,
        "velocity_error": REWARD_DT
        * REWARD_SCALES["velocity_error"]
        * linear_error,
        "tracking_ang_vel": REWARD_DT
        * REWARD_SCALES["tracking_ang_vel"]
        * tracking_ang_vel,
        "lin_vel_z": REWARD_DT * REWARD_SCALES["lin_vel_z"] * lin_vel_z,
        "base_height": REWARD_DT
        * REWARD_SCALES["base_height"]
        * base_height_error,
        "action_rate": REWARD_DT
        * REWARD_SCALES["action_rate"]
        * action_rate,
        "similar_to_default": REWARD_DT
        * REWARD_SCALES["similar_to_default"]
        * similar_to_default,
    }
    return RewardTerms(**weighted, total=sum(weighted.values()))
