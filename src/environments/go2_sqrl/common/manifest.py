"""Go2-specific checkpoint contract construction and validation."""

from __future__ import annotations

from typing import Any

from .reward import (
    BASE_HEIGHT_TARGET,
    REWARD_DT,
    REWARD_DEFAULT_JOINT_POSITION,
    REWARD_SCALES,
    REWARD_VERSION,
    TRACKING_SIGMA,
)
from .specs import (
    ACTION_SPEC,
    DEFAULT_BASE_HEIGHT,
    OBSERVATION_SPEC,
    PHYSICS_DT,
)


MANIFEST_VERSION = 8
VELOCITY_ESTIMATOR_VERSION = "contact-free-robust-kf-v1"
FAILURE_CONTRACT_VERSION = "imu-roll-pitch-sustained-v1"
ACTION_PIPELINE_VERSION = "sdk-absolute-position-v2"


def build_manifest(
    normalizer: dict[str, Any] | None = None,
    *,
    fall_angle_threshold: float = 0.8,
    fall_consecutive_frames: int = 5,
    target_velocity_x: float = 0.5,
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "observation": {
            "version": OBSERVATION_SPEC.version,
            "size": OBSERVATION_SPEC.size,
            "joint_order": list(OBSERVATION_SPEC.joint_order),
            "quaternion_order": OBSERVATION_SPEC.quaternion_order,
            "body_velocity": {
                "indices": [
                    OBSERVATION_SPEC.body_velocity.start,
                    OBSERVATION_SPEC.body_velocity.stop,
                ],
                "frame": "body",
                "source": "proprioceptive_velocity_estimator",
            },
            "velocity_estimator": {
                "version": VELOCITY_ESTIMATOR_VERSION,
                "policy_visible": True,
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
            "pipeline_version": ACTION_PIPELINE_VERSION,
            "size": ACTION_SPEC.size,
            "joint_order": list(ACTION_SPEC.joint_order),
            "scale": ACTION_SPEC.scale,
            "default_position": list(ACTION_SPEC.default_position),
            "target_semantics": "absolute_joint_position",
            "backend_offset_semantics": "shared_default_position",
            "max_target_rate": ACTION_SPEC.max_target_rate,
            "kp": ACTION_SPEC.kp,
            "kd": ACTION_SPEC.kd,
            "effort_limit": ACTION_SPEC.effort_limit,
            "velocity_limit": ACTION_SPEC.velocity_limit,
            "armature": ACTION_SPEC.armature,
            "joint_damping": ACTION_SPEC.joint_damping,
            "joint_friction": ACTION_SPEC.joint_friction,
            "control_dt": ACTION_SPEC.control_dt,
        },
        "reset": {
            "joint_position": list(ACTION_SPEC.default_position),
            "base_height": DEFAULT_BASE_HEIGHT,
        },
        "reward_version": REWARD_VERSION,
        "reward_contract": {
            "source": (
                "https://github.com/Holiday-Robot/FlashSAC/blob/main/"
                "flash_rl/envs/genesis_envs/go2_walk_easy.py"
            ),
            "dt": REWARD_DT,
            "tracking_sigma": TRACKING_SIGMA,
            "base_height_target": BASE_HEIGHT_TARGET,
            "reward_scales_before_dt": dict(REWARD_SCALES),
            "similar_to_default_joint_position": list(
                REWARD_DEFAULT_JOINT_POSITION.tolist()
            ),
            "linear_velocity_frame": "full_quaternion_body_frame",
            "command": {
                "linear_velocity_x": float(target_velocity_x),
                "linear_velocity_y": 0.0,
                "angular_velocity_z": 0.0,
            },
            "failure_reward_shaping": False,
            "stationary_local_optimum_fix": (
                "negative_squared_xy_velocity_command_error"
            ),
        },
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
