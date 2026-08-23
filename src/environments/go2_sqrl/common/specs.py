"""Versioned Go2 observation and action contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np


JOINT_NAMES: Final[tuple[str, ...]] = tuple(
    f"{leg}_{joint}"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
OBSERVATION_SIZE: Final = 46
ACTION_SIZE: Final = 12
CONTROL_DT: Final = 0.02
PHYSICS_DT: Final = 0.002
PHYSICS_STEPS_PER_ACTION: Final = 10
EPISODE_STEPS: Final = 500

DEFAULT_JOINT_POSITION = np.tile(
    np.asarray([0.0, 0.9, -1.8], dtype=np.float32), 4
)
JOINT_LOWER_LIMIT = np.asarray(
    [
        -1.0472, -1.5708, -2.7227,
        -1.0472, -1.5708, -2.7227,
        -1.0472, -0.5236, -2.7227,
        -1.0472, -0.5236, -2.7227,
    ],
    dtype=np.float32,
)
JOINT_UPPER_LIMIT = np.asarray(
    [
        1.0472, 3.4907, -0.83776,
        1.0472, 3.4907, -0.83776,
        1.0472, 4.5379, -0.83776,
        1.0472, 4.5379, -0.83776,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ObservationSpecV1:
    version: str = "go2-observation-v1"
    size: int = OBSERVATION_SIZE
    joint_q: slice = field(default_factory=lambda: slice(0, 12))
    joint_dq: slice = field(default_factory=lambda: slice(12, 24))
    imu_gyro: slice = field(default_factory=lambda: slice(24, 27))
    body_velocity: slice = field(default_factory=lambda: slice(27, 30))
    imu_quat: slice = field(default_factory=lambda: slice(30, 34))
    previous_action_q_target: slice = field(default_factory=lambda: slice(34, 46))
    quaternion_order: str = "WXYZ"
    joint_order: tuple[str, ...] = JOINT_NAMES


@dataclass(frozen=True)
class ActionSpecV1:
    version: str = "go2-action-v1"
    size: int = ACTION_SIZE
    scale: float = 0.25
    control_dt: float = CONTROL_DT
    max_target_rate: float = 12.0
    kp: float = 25.0
    kd: float = 0.5
    default_position: tuple[float, ...] = tuple(DEFAULT_JOINT_POSITION.tolist())
    joint_order: tuple[str, ...] = JOINT_NAMES


OBSERVATION_SPEC = ObservationSpecV1()
ACTION_SPEC = ActionSpecV1()


def joint_order_indices(source_names: list[str] | tuple[str, ...]) -> np.ndarray:
    """Return gather indices that convert ``source_names`` into SDK order."""

    normalized = [name.removesuffix("_joint") for name in source_names]
    missing = [name for name in JOINT_NAMES if name not in normalized]
    if missing:
        raise ValueError(f"Missing Go2 joints: {missing}")
    return np.asarray([normalized.index(name) for name in JOINT_NAMES], dtype=np.int64)
