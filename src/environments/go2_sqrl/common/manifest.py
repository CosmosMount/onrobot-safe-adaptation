"""Go2-specific checkpoint contract construction and validation."""

from __future__ import annotations

from typing import Any

from .reward import REWARD_VERSION
from .specs import ACTION_SPEC, OBSERVATION_SPEC, PHYSICS_DT


MANIFEST_VERSION = 2
VELOCITY_ESTIMATOR_VERSION = "proprioceptive-support-v2"
FAILURE_CONTRACT_VERSION = "imu-roll-pitch-sustained-v1"


def build_manifest(
    normalizer: dict[str, Any] | None = None,
    *,
    fall_angle_threshold: float = 0.8,
    fall_consecutive_frames: int = 5,
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "observation": {
            "version": OBSERVATION_SPEC.version,
            "size": OBSERVATION_SPEC.size,
            "joint_order": list(OBSERVATION_SPEC.joint_order),
            "quaternion_order": OBSERVATION_SPEC.quaternion_order,
            "velocity_estimator": {
                "version": VELOCITY_ESTIMATOR_VERSION,
                "inputs": [
                    "joint_q",
                    "joint_dq",
                    "imu_gyro",
                    "imu_quat",
                    "imu_accelerometer",
                ],
                "external_contact_sensor": False,
            },
        },
        "action": {
            "version": ACTION_SPEC.version,
            "size": ACTION_SPEC.size,
            "scale": ACTION_SPEC.scale,
            "max_target_rate": ACTION_SPEC.max_target_rate,
            "kp": ACTION_SPEC.kp,
            "kd": ACTION_SPEC.kd,
            "control_dt": ACTION_SPEC.control_dt,
        },
        "reward_version": REWARD_VERSION,
        "failure": {
            "version": FAILURE_CONTRACT_VERSION,
            "signal": "imu_quaternion_roll_pitch",
            "angle_threshold": float(fall_angle_threshold),
            "consecutive_frames": int(fall_consecutive_frames),
            "frame_unit": "physics_frames",
            "frame_dt": PHYSICS_DT,
            "external_contact_sensor": False,
        },
        "physics_dt": PHYSICS_DT,
        "normalizer": normalizer,
    }


def validate_manifest(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    def compare(path: str, lhs: Any, rhs: Any) -> None:
        if isinstance(rhs, dict):
            if not isinstance(lhs, dict):
                raise ValueError(f"Checkpoint contract mismatch at {path}")
            for key, value in rhs.items():
                if key not in lhs:
                    raise ValueError(f"Checkpoint contract missing {path}{key}")
                compare(f"{path}{key}.", lhs[key], value)
        elif lhs != rhs:
            raise ValueError(
                f"Checkpoint contract mismatch at {path[:-1]}: expected {rhs}, got {lhs}"
            )

    compare("", actual, expected)
