"""Checkpoint manifest construction and validation."""

from dataclasses import asdict
from typing import Any

from .contracts import (
    ACTION_SPEC,
    CONTACT_FRICTION,
    CONTROL_DT,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_BASE_QUATERNION_WXYZ,
    EPISODE_STEPS,
    FAILURE_SPEC,
    GRAVITY_Z,
    OBSERVATION_SPEC,
    PHYSICS_DT,
)
from .estimation import (
    DEFAULT_VELOCITY_ESTIMATOR_CONFIG,
    VelocityEstimatorConfig,
)


def build_manifest(
    normalizer: dict[str, Any] | None = None,
    *,
    fall_angle_threshold: float = FAILURE_SPEC.angle_threshold,
    fall_min_base_clearance: float = FAILURE_SPEC.min_base_clearance,
    fall_consecutive_frames: int = FAILURE_SPEC.consecutive_frames,
    target_velocity_x: float = 0.5,
    domain_randomization: bool = False,
    velocity_estimator_config: VelocityEstimatorConfig = (
        DEFAULT_VELOCITY_ESTIMATOR_CONFIG
    ),
) -> dict[str, Any]:
    return {
        "environment": {
            "terrain": "flat",
            "domain_randomization": bool(domain_randomization),
            "friction": CONTACT_FRICTION,
            "gravity_z": GRAVITY_Z,
            "physics_dt": PHYSICS_DT,
            "control_dt": CONTROL_DT,
            "episode_steps": EPISODE_STEPS,
            "target_velocity_x": float(target_velocity_x),
        },
        "observation": {
            "size": OBSERVATION_SPEC.size,
            "joint_order": list(OBSERVATION_SPEC.joint_order),
            "quaternion_order": OBSERVATION_SPEC.quaternion_order,
            "layout": {
                "joint_q": [
                    OBSERVATION_SPEC.joint_q.start,
                    OBSERVATION_SPEC.joint_q.stop,
                ],
                "joint_dq": [
                    OBSERVATION_SPEC.joint_dq.start,
                    OBSERVATION_SPEC.joint_dq.stop,
                ],
                "imu_gyro": [
                    OBSERVATION_SPEC.imu_gyro.start,
                    OBSERVATION_SPEC.imu_gyro.stop,
                ],
                "body_velocity": [
                    OBSERVATION_SPEC.body_velocity.start,
                    OBSERVATION_SPEC.body_velocity.stop,
                ],
                "imu_quat": [
                    OBSERVATION_SPEC.imu_quat.start,
                    OBSERVATION_SPEC.imu_quat.stop,
                ],
                "previous_action_q_target": [
                    OBSERVATION_SPEC.previous_action_q_target.start,
                    OBSERVATION_SPEC.previous_action_q_target.stop,
                ],
            },
            "body_velocity": {
                "indices": [
                    OBSERVATION_SPEC.body_velocity.start,
                    OBSERVATION_SPEC.body_velocity.stop,
                ],
                "frame": "body",
                "source": "proprioceptive_velocity_estimator",
            },
            "velocity_estimator": {
                "policy_visible": True,
                "inputs": [
                    "joint_q",
                    "joint_dq",
                    "imu_gyro",
                    "imu_quat",
                    "imu_accelerometer",
                ],
                "external_contact_sensor": False,
                "parameters": asdict(velocity_estimator_config),
            },
        },
        "action": {
            "size": ACTION_SPEC.size,
            "joint_order": list(ACTION_SPEC.joint_order),
            "scale": list(ACTION_SPEC.scale),
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
            "base_quaternion_wxyz": list(DEFAULT_BASE_QUATERNION_WXYZ),
            "base_height_reference": "flat_ground_surface",
            "nominal_foot_contact": "all_four_feet",
        },
        "reward_contract": {
            "source": "https://arxiv.org/abs/2503.08375",
            "name": "r_total-track-x",
            "time_scaling": "none",
            "tracking_velocity": (
                "1 in [target, 2*target]; 0 at <=-target or >=4*target; "
                "otherwise 1-|vx-target|/(2*target)"
            ),
            "linear_velocity_frame": "body",
            "penalty_scales": {
                "yaw_rate_squared": -0.1,
                "roll_pitch_squared": -10.0,
                "joint_torque_squared": -0.0003,
            },
            "clip": "max(sum, 0)",
            "command": {
                "linear_velocity_x": float(target_velocity_x),
                "linear_velocity_y": 0.0,
                "angular_velocity_z": 0.0,
            },
            "failure_reward_shaping": False,
        },
        "failure": {
            "signal": [
                "imu_quaternion_roll_pitch",
                "base_clearance_above_local_terrain",
            ],
            "aggregation": "tilt_or_low_base",
            "angle_threshold": float(fall_angle_threshold),
            "min_base_clearance": float(fall_min_base_clearance),
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


def validate_transfer_manifest(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Validate shared semantics while allowing intended phase differences."""

    actual = {
        **actual,
        "environment": {
            key: value for key, value in actual.get("environment", {}).items()
            if key not in ("domain_randomization", "target_velocity_x")
        },
        "reward_contract": {
            key: value for key, value in actual.get("reward_contract", {}).items()
            if key != "command"
        },
    }
    expected = {
        **expected,
        "environment": {
            key: value for key, value in expected.get("environment", {}).items()
            if key not in ("domain_randomization", "target_velocity_x")
        },
        "reward_contract": {
            key: value for key, value in expected.get("reward_contract", {}).items()
            if key != "command"
        },
    }
    validate_manifest(actual, expected)

