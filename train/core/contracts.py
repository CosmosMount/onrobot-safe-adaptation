"""Shared Go2 policy, physics and environment contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

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
DEFAULT_BASE_HEIGHT: Final = 0.289
DEFAULT_BASE_QUATERNION_WXYZ: Final[tuple[float, ...]] = (1.0, 0.0, 0.0, 0.0)
TARGET_VELOCITY_X: Final = 0.5
CONTACT_FRICTION: Final = 0.4
GRAVITY_Z: Final = -9.81

DEFAULT_JOINT_POSITION = np.tile(
    np.asarray([0.0, 0.9, -1.8], dtype=np.float32), 4
)
ACTION_SCALE = np.tile(
    # Preserve conservative lateral hip motion while giving the sagittal
    # joints enough range for roughly 10 cm of useful swing-foot clearance.
    np.asarray([0.25, 0.35, 0.45], dtype=np.float32), 4
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
class ObservationSpec:
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
class ActionSpec:
    size: int = ACTION_SIZE
    scale: tuple[float, ...] = tuple(ACTION_SCALE.tolist())
    control_dt: float = CONTROL_DT
    max_target_rate: float = 12.0
    # Training/checkpoint actuator contract. MuJoCo-only PD sensitivity
    # experiments use environment.policy_kp/policy_kd instead.
    kp: float = 25.0
    kd: float = 0.5
    effort_limit: float = 23.5
    velocity_limit: float = 1000.0
    armature: float = 0.0
    joint_damping: float = 0.0
    joint_friction: float = 0.0
    default_position: tuple[float, ...] = tuple(DEFAULT_JOINT_POSITION.tolist())
    joint_order: tuple[str, ...] = JOINT_NAMES


@dataclass(frozen=True)
class FailureSpec:
    """Sparse SQRL incident label shared by source and target backends."""

    angle_threshold: float = 0.8
    min_base_clearance: float = 0.18
    consecutive_frames: int = 5


# Canonical source/target tensor contract.
OBSERVATION_SPEC = ObservationSpec()
ACTION_SPEC = ActionSpec()
FAILURE_SPEC = FailureSpec()


def configure_failure_detection(config) -> None:
    """Expose the exact common SQRL failure label on an environment config."""

    config.fall_angle_threshold = FAILURE_SPEC.angle_threshold
    config.fall_min_base_clearance = FAILURE_SPEC.min_base_clearance
    config.fall_consecutive_frames = FAILURE_SPEC.consecutive_frames


def configure_environment_contract(config) -> None:
    """Apply the nominal settings shared by Isaac Lab and MuJoCo."""

    config.target_velocity_x = TARGET_VELOCITY_X
    config.terrain_mode = "flat"
    config.domain_randomization = False
    config.friction = CONTACT_FRICTION
    config.gravity_z = GRAVITY_Z
    config.reset_base_height = DEFAULT_BASE_HEIGHT
    config.physics_dt = PHYSICS_DT
    config.control_dt = CONTROL_DT
    config.policy_frames = PHYSICS_STEPS_PER_ACTION
    config.policy_period_seconds = CONTROL_DT
    config.episode_steps = EPISODE_STEPS
    config.policy_kp = ACTION_SPEC.kp
    config.policy_kd = ACTION_SPEC.kd
    configure_failure_detection(config)


def validate_environment_contract(config) -> None:
    """Validate shared physics while allowing intentional phase settings."""

    expected = {
        "terrain_mode": "flat",
        "friction": CONTACT_FRICTION,
        "gravity_z": GRAVITY_Z,
        "reset_base_height": DEFAULT_BASE_HEIGHT,
        "physics_dt": PHYSICS_DT,
        "control_dt": CONTROL_DT,
        "policy_frames": PHYSICS_STEPS_PER_ACTION,
        "policy_period_seconds": CONTROL_DT,
        "episode_steps": EPISODE_STEPS,
        "policy_kp": ACTION_SPEC.kp,
        "policy_kd": ACTION_SPEC.kd,
        "fall_angle_threshold": FAILURE_SPEC.angle_threshold,
        "fall_min_base_clearance": FAILURE_SPEC.min_base_clearance,
        "fall_consecutive_frames": FAILURE_SPEC.consecutive_frames,
    }
    for name, value in expected.items():
        actual = getattr(config, name, None)
        if isinstance(value, float):
            aligned = actual is not None and np.isclose(float(actual), value, rtol=0.0, atol=1e-12)
        else:
            aligned = actual == value
        if not aligned:
            raise ValueError(f"environment.{name} must be {value!r}, got {actual!r}")
    target_velocity_x = getattr(config, "target_velocity_x", None)
    if target_velocity_x is None or not np.isfinite(float(target_velocity_x)) or float(target_velocity_x) <= 0.0:
        raise ValueError(f"environment.target_velocity_x must be positive, got {target_velocity_x!r}")
    domain_randomization = getattr(config, "domain_randomization", None)
    if not isinstance(domain_randomization, (bool, np.bool_)):
        raise ValueError(
            "environment.domain_randomization must be a boolean, "
            f"got {domain_randomization!r}"
        )


def format_policy_io_contract(target_velocity_x: float) -> str:
    """Format the tensors exchanged with the policy."""

    rows = (
        ("joint_q", OBSERVATION_SPEC.joint_q, "measured joint position"),
        ("joint_dq", OBSERVATION_SPEC.joint_dq, "measured joint velocity"),
        ("imu_gyro", OBSERVATION_SPEC.imu_gyro, "body-frame IMU angular velocity"),
        (
            "body_velocity",
            OBSERVATION_SPEC.body_velocity,
            "robust IMU + leg-odometry estimate in the body frame",
        ),
        ("imu_quat", OBSERVATION_SPEC.imu_quat, "continuous WXYZ IMU quaternion"),
        (
            "previous_action_q_target",
            OBSERVATION_SPEC.previous_action_q_target,
            "previous applied joint-position target",
        ),
    )
    lines = [
        "[INFO] Go2 policy I/O (actual model tensors)",
        f"  Policy observation: shape ({OBSERVATION_SPEC.size},)",
    ]
    for name, indices, source in rows:
        lines.append(
            f"    [{indices.start:>2}:{indices.stop:<2}] {name:<31} {source}"
        )
    lines.extend(
        (
            "  velocity_commands: not observed by the policy",
            "  simulator base_lin_vel: reward and diagnostics only when available",
            "  reward velocity fallback: robust body_velocity estimate",
            f"  reward target_velocity_x: {float(target_velocity_x):g} m/s",
            f"  Policy action: shape ({ACTION_SPEC.size},), normalized joint-position targets",
        )
    )
    return "\n".join(lines)

# Backend-neutral state and transition records.
Array = Any


@dataclass(slots=True)
class RobotState:
    joint_q: Array
    joint_dq: Array
    imu_gyro: Array
    imu_quat: Array
    imu_accelerometer: Array | None = None
    actuator_torque: Array | None = None
    tick: int | None = None
    # Nonzero only when an external simulator proves this is a completed
    # post-step snapshot for the matching LowCmd transaction.
    command_sequence: int | None = None


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
    tracking_velocity: float
    yaw_rate: float
    upright: float
    energy: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reward/tracking_velocity": self.tracking_velocity,
            "reward/yaw_rate": self.yaw_rate,
            "reward/upright": self.upright,
            "reward/energy": self.energy,
            "reward/total": self.total,
        }

