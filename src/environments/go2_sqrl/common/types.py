"""Data records separating policy-visible state from training-only truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np


Array = Any


@dataclass(slots=True)
class RobotState:
    joint_q: Array
    joint_dq: Array
    imu_gyro: Array
    imu_quat: Array
    imu_accelerometer: Array | None = None
    tick: int | None = None


@dataclass(slots=True)
class TrainingState:
    world_velocity: Array
    base_position: Array | None = None
    actuator_torque: Array | None = None


@dataclass(slots=True)
class ActionResult:
    raw_action: np.ndarray
    applied_action: np.ndarray
    q_target: np.ndarray


@dataclass(slots=True)
class RewardTerms:
    track_x: float
    track_xy: float
    yaw: float
    upright: float
    energy: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reward/track_x": self.track_x,
            "reward/track_xy": self.track_xy,
            "reward/yaw": self.yaw,
            "reward/upright": self.upright,
            "reward/energy": self.energy,
            "reward/total": self.total,
        }


class StepInfo(TypedDict, total=False):
    failure: np.ndarray
    applied_action: np.ndarray
    velocity_estimation_error: np.ndarray
    episode_return: np.ndarray
    episode_length: np.ndarray
    final_observation: list[np.ndarray | None]
    final_info: list[dict[str, Any] | None]
