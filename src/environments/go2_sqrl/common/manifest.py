"""Go2-specific checkpoint contract construction and validation."""

from __future__ import annotations

from typing import Any

from .reward import (
    BASE_HEIGHT_TARGET,
    FOOT_CLEARANCE_TARGET,
    HIGH_CLEARANCE_REWARD_VERSION,
    REWARD_DT,
    REWARD_DEFAULT_JOINT_POSITION,
    REWARD_SCALES,
    REWARD_VERSION,
    PHASE_REFERENCE_FREQUENCY,
    TRACKING_SIGMA,
)
from .specs import (
    ACTION_SPEC,
    DEFAULT_ACTION_PROFILE,
    DEFAULT_BASE_HEIGHT,
    FAILURE_SPEC,
    OBSERVATION_SPEC,
    PHYSICS_DT,
    action_profile as resolve_action_profile,
)


MANIFEST_VERSION = 11
VELOCITY_ESTIMATOR_VERSION = "contact-free-robust-kf-v1"
FAILURE_CONTRACT_VERSION = FAILURE_SPEC.version
ACTION_PIPELINE_VERSION = "sdk-absolute-position-v3-per-joint-scale"


def build_manifest(
    normalizer: dict[str, Any] | None = None,
    *,
    fall_angle_threshold: float = FAILURE_SPEC.angle_threshold,
    fall_min_base_clearance: float = FAILURE_SPEC.min_base_clearance,
    fall_consecutive_frames: int = FAILURE_SPEC.consecutive_frames,
    target_velocity_x: float = 0.5,
    foot_clearance_target: float = FOOT_CLEARANCE_TARGET,
    phase_reference_frequency: float = PHASE_REFERENCE_FREQUENCY,
    phase_reward_scale: float = 0.0,
    clearance_reward_mode: str = "swing_weighted",
    foot_clearance_upper_target: float = 0.0,
    foot_clearance_overshoot_scale: float = 0.0,
    phase_velocity_gate_start: float = 0.0,
    phase_velocity_gate_full: float = 0.0,
    stable_progress_start: float = 1.0,
    stable_progress_min_base_clearance: float = 0.22,
    stable_progress_scale: float = 0.0,
    terminal_failure_penalty: float = 0.0,
    action_profile: str = DEFAULT_ACTION_PROFILE,
    foot_clearance_reward_scale: float = REWARD_SCALES["foot_clearance"],
) -> dict[str, Any]:
    action_contract = resolve_action_profile(action_profile)
    high_clearance_enabled = bool(
        float(foot_clearance_upper_target) > 0.0
        or float(foot_clearance_overshoot_scale) != 0.0
        or float(phase_velocity_gate_full) > float(phase_velocity_gate_start)
        or float(stable_progress_scale) != 0.0
    )
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
            "version": action_contract["version"],
            "pipeline_version": action_contract["pipeline_version"],
            "size": ACTION_SPEC.size,
            "joint_order": list(ACTION_SPEC.joint_order),
            "scale": (
                float(action_contract["scale"][0])
                if str(action_profile) == "legacy_v1"
                else action_contract["scale"].tolist()
            ),
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
        "reward_version": (
            HIGH_CLEARANCE_REWARD_VERSION
            if high_clearance_enabled
            else REWARD_VERSION
        ),
        "reward_contract": {
            "source": (
                "https://github.com/Holiday-Robot/FlashSAC/blob/main/"
                "flash_rl/envs/genesis_envs/go2_walk_easy.py"
            ),
            "dt": REWARD_DT,
            "tracking_sigma": TRACKING_SIGMA,
            "base_height_target": BASE_HEIGHT_TARGET,
            "base_height_reference": "local_terrain_clearance",
            "foot_clearance_target": float(foot_clearance_target),
            "foot_clearance_reference": "local_terrain_under_each_foot",
            "foot_swing_signal": "world_horizontal_foot_speed",
            "foot_clearance_aggregation": str(clearance_reward_mode),
            "phase": {
                "source": "foot_clearance_and_vertical_velocity",
                "leg_order": ["FR", "FL", "RR", "RL"],
                "reference_frequency": float(phase_reference_frequency),
                "reward_scale_before_dt": float(phase_reward_scale),
                "external_clock": False,
                "policy_observation": False,
            },
            "reward_scales_before_dt": {
                **dict(REWARD_SCALES),
                "foot_clearance": float(foot_clearance_reward_scale),
                "phase": float(phase_reward_scale),
            },
            "similar_to_default_joint_position": list(
                REWARD_DEFAULT_JOINT_POSITION.tolist()
            ),
            "linear_velocity_frame": "full_quaternion_body_frame",
            "command": {
                "linear_velocity_x": float(target_velocity_x),
                "linear_velocity_y": 0.0,
                "angular_velocity_z": 0.0,
            },
            "failure_reward_shaping": bool(float(terminal_failure_penalty) != 0.0),
            **(
                {"terminal_failure_penalty": float(terminal_failure_penalty)}
                if float(terminal_failure_penalty) != 0.0
                else {}
            ),
            "stationary_local_optimum_fix": (
                "negative_squared_xy_velocity_command_error"
            ),
            **(
                {
                    "high_clearance": {
                        "upper_target": float(foot_clearance_upper_target),
                        "overshoot_scale_before_dt": float(
                            foot_clearance_overshoot_scale
                        ),
                        "phase_velocity_gate": [
                            float(phase_velocity_gate_start),
                            float(phase_velocity_gate_full),
                        ],
                        "stable_progress_start": float(stable_progress_start),
                        "stable_progress_min_base_clearance": float(
                            stable_progress_min_base_clearance
                        ),
                        "stable_progress_scale_before_dt": float(
                            stable_progress_scale
                        ),
                        "policy_observation_changed": False,
                    }
                }
                if high_clearance_enabled
                else {}
            ),
        },
        "failure": {
            "version": FAILURE_CONTRACT_VERSION,
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
    """Validate policy/QSafe compatibility while allowing a new task reward.

    Changing the reward or velocity command is the defining SQRL fine-tuning
    operation.  Policy tensor meaning, safety labels, action/reset semantics,
    and normalization remain strict transfer invariants.
    """

    transfer_expected = {
        key: value
        for key, value in expected.items()
        if key not in ("reward_version", "reward_contract")
    }
    validate_manifest(actual, transfer_expected)


def validate_actor_transfer_manifest(
    actual: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Validate actor-visible semantics while allowing a new target task.

    A SAC actor does not consume reward or failure labels. This permits the
    proven manifest-v9 actor to use the repaired relative-ground failure
    detector while observation, action mapping, reset, timing and normalizer
    remain exact.
    """

    actor_expected = {
        key: expected[key]
        for key in (
            "observation",
            "action",
            "reset",
            "physics_dt",
            "normalizer",
        )
    }
    validate_manifest(actual, actor_expected)
