"""Shared Go2 reward and episode accounting."""

import numpy as np

from .contracts import RewardTerms
from .observation import quaternion_to_rpy_wxyz


REWARD_SCALES = {
    "tracking_velocity": 1.0,
    "yaw_rate": -0.1,
    "upright": -10.0,
    "energy": -0.0003,
}


def track_x_reward(velocity_x: float, target_velocity_x: float) -> float:
    """Piecewise forward-velocity tracking term from Gait in Eight."""

    velocity_x = float(velocity_x)
    target_velocity_x = float(target_velocity_x)
    if target_velocity_x <= 0.0:
        raise ValueError("target_velocity_x must be positive")
    if target_velocity_x <= velocity_x <= 2.0 * target_velocity_x:
        return 1.0
    if velocity_x <= -target_velocity_x or velocity_x >= 4.0 * target_velocity_x:
        return 0.0
    return 1.0 - abs(velocity_x - target_velocity_x) / (2.0 * target_velocity_x)


def compute_reward(
    body_velocity: np.ndarray,
    imu_quat: np.ndarray,
    imu_gyro: np.ndarray,
    actuator_torque: np.ndarray,
    target_velocity_x: float,
) -> RewardTerms:
    """Compute Gait in Eight's non-negative fixed-forward tracking reward."""

    roll, pitch, _ = quaternion_to_rpy_wxyz(imu_quat)
    raw = {
        "tracking_velocity": REWARD_SCALES["tracking_velocity"]
        * track_x_reward(np.asarray(body_velocity, dtype=np.float64)[0], target_velocity_x),
        "yaw_rate": REWARD_SCALES["yaw_rate"]
        * float(np.asarray(imu_gyro, dtype=np.float64)[2] ** 2),
        "upright": REWARD_SCALES["upright"] * float(roll * roll + pitch * pitch),
        "energy": REWARD_SCALES["energy"]
        * float(np.square(np.asarray(actuator_torque, dtype=np.float64)).sum()),
    }
    return RewardTerms(**raw, total=max(sum(raw.values()), 0.0))


def compute_reward_tensor(
    body_velocity,
    imu_quat,
    imu_gyro,
    actuator_torque,
    target_velocity_x: float,
):
    """Vectorized PyTorch form of :func:`compute_reward` for Isaac Lab."""

    import torch

    target = float(target_velocity_x)
    if target <= 0.0:
        raise ValueError("target_velocity_x must be positive")
    velocity_x = body_velocity[:, 0]
    tracking = 1.0 - (velocity_x - target).abs() / (2.0 * target)
    tracking = torch.where(
        (velocity_x >= target) & (velocity_x <= 2.0 * target),
        torch.ones_like(tracking),
        tracking,
    )
    tracking = torch.where(
        (velocity_x <= -target) | (velocity_x >= 4.0 * target),
        torch.zeros_like(tracking),
        tracking,
    )
    quaternion = imu_quat / torch.linalg.vector_norm(
        imu_quat, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(dim=-1)
    roll = torch.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x.square() + y.square()),
    )
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    terms = {
        "tracking_velocity": REWARD_SCALES["tracking_velocity"] * tracking,
        "yaw_rate": REWARD_SCALES["yaw_rate"] * imu_gyro[:, 2].square(),
        "upright": REWARD_SCALES["upright"]
        * (roll.square() + pitch.square()),
        "energy": REWARD_SCALES["energy"]
        * actuator_torque.square().sum(dim=-1),
    }
    total = torch.stack(tuple(terms.values()), dim=0).sum(dim=0).clamp_min(0.0)
    return terms, total


class EpisodeTracker:
    def __init__(self, max_steps: int = 500):
        self.max_steps = int(max_steps)
        self.steps = 0
        self.episode_return = 0.0

    def reset(self) -> None:
        self.steps = 0
        self.episode_return = 0.0

    def advance(self, reward: float, failure: bool) -> tuple[bool, bool]:
        self.steps += 1
        self.episode_return += float(reward)
        terminated = bool(failure)
        truncated = self.steps >= self.max_steps and not terminated
        return terminated, truncated

