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
    tracking_lin_vel: float
    velocity_error: float
    tracking_ang_vel: float
    lin_vel_z: float
    base_height: float
    foot_clearance: float
    foot_clearance_overshoot: float
    phase: float
    stable_progress: float
    action_rate: float
    similar_to_default: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reward/tracking_lin_vel": self.tracking_lin_vel,
            "reward/velocity_error": self.velocity_error,
            "reward/tracking_ang_vel": self.tracking_ang_vel,
            "reward/lin_vel_z": self.lin_vel_z,
            "reward/base_height": self.base_height,
            "reward/foot_clearance": self.foot_clearance,
            "reward/foot_clearance_overshoot": self.foot_clearance_overshoot,
            "reward/phase": self.phase,
            "reward/stable_progress": self.stable_progress,
            "reward/action_rate": self.action_rate,
            "reward/similar_to_default": self.similar_to_default,
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
