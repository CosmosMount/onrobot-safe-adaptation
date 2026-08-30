"""Shared Go2 observation, reward, termination, and checkpoint contracts."""
from __future__ import annotations

from typing import Protocol, Sequence

from .base import (
    ACTION_SPEC, DEFAULT_BASE_HEIGHT, DEFAULT_JOINT_POSITION, FAILURE_SPEC,
    OBSERVATION_SPEC, PHYSICS_DT, RewardTerms, RobotState,
)
from .estimation import VelocityEstimator

def continuous_quaternion_wxyz(
    quaternion: np.ndarray, previous: np.ndarray | None
) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32).copy()
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid IMU quaternion")
    quaternion /= norm
    if previous is None:
        if quaternion[0] < 0:
            quaternion *= -1
    elif float(np.dot(quaternion, previous)) < 0:
        quaternion *= -1
    return quaternion


def build_observation(
    state: RobotState,
    estimated_body_velocity: np.ndarray,
    previous_q_target: np.ndarray,
    previous_quaternion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    quaternion = continuous_quaternion_wxyz(state.imu_quat, previous_quaternion)
    observation = np.empty(OBSERVATION_SPEC.size, dtype=np.float32)
    observation[OBSERVATION_SPEC.joint_q] = np.asarray(state.joint_q, dtype=np.float32)
    observation[OBSERVATION_SPEC.joint_dq] = np.asarray(state.joint_dq, dtype=np.float32)
    observation[OBSERVATION_SPEC.imu_gyro] = np.asarray(
        state.imu_gyro, dtype=np.float32
    )
    observation[OBSERVATION_SPEC.body_velocity] = np.asarray(
        estimated_body_velocity, dtype=np.float32
    )
    observation[OBSERVATION_SPEC.imu_quat] = quaternion
    observation[OBSERVATION_SPEC.previous_action_q_target] = np.asarray(
        previous_q_target, dtype=np.float32
    )
    if not np.all(np.isfinite(observation)):
        raise ValueError("Observation contains NaN or infinity")
    return observation, quaternion


class BodyVelocityEstimator(Protocol):
    def reset(self) -> None: ...

    def update(self, state: RobotState) -> np.ndarray: ...


# Shared observation, reward and episode accounting.
class ObservationBuilder:
    def __init__(
        self, velocity_estimator: BodyVelocityEstimator | None = None
    ):
        self.velocity_estimator = velocity_estimator or VelocityEstimator()
        self.previous_quaternion: np.ndarray | None = None
        self.previous_q_target = DEFAULT_JOINT_POSITION.copy()

    def reset(self, previous_q_target: np.ndarray | None = None) -> None:
        self.velocity_estimator.reset()
        self.previous_quaternion = None
        self.previous_q_target = np.asarray(
            DEFAULT_JOINT_POSITION if previous_q_target is None else previous_q_target,
            dtype=np.float32,
        ).copy()

    def set_previous_q_target(self, q_target: np.ndarray) -> None:
        self.previous_q_target = np.asarray(q_target, dtype=np.float32).copy()

    def _build_with_velocity(
        self, state: RobotState, estimated_body_velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        observation, quaternion = build_observation(
            state,
            estimated_body_velocity,
            self.previous_q_target,
            self.previous_quaternion,
        )
        self.previous_quaternion = quaternion
        return observation, estimated_body_velocity

    def build(self, state: RobotState) -> tuple[np.ndarray, np.ndarray]:
        estimated_body_velocity = self.velocity_estimator.update(state)
        return self._build_with_velocity(state, estimated_body_velocity)

    def build_many(
        self, states: Sequence[RobotState]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance every physics frame and build one observation from the last."""

        if len(states) == 0:
            raise ValueError("states must contain at least one physics frame")
        estimated_body_velocity = None
        for state in states:
            estimated_body_velocity = self.velocity_estimator.update(state)
        return self._build_with_velocity(states[-1], estimated_body_velocity)

import math

import numpy as np


REWARD_VERSION = "flashsac-go2-walk-easy-command-v5-swing-clearance"
REWARD_DT = 0.02
TRACKING_SIGMA = 0.25
BASE_HEIGHT_TARGET = 0.3
FOOT_CLEARANCE_TARGET = 0.10
SWING_SPEED_START = 0.15
SWING_SPEED_FULL = 0.50
REWARD_SCALES = {
    "tracking_lin_vel": 1.0,
    # The source exponential still pays 36.8% of its maximum at zero speed
    # for a 0.5 m/s command.  This explicit error cost makes standing still a
    # negative-return local optimum while leaving exact tracking unchanged.
    "velocity_error": -3.0,
    "tracking_ang_vel": 0.2,
    "lin_vel_z": -1.0,
    "base_height": -50.0,
    "foot_clearance": -20.0,
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


def local_base_clearance(base_world_height, local_ground_world_height):
    """Return base height measured from the local terrain surface."""

    return base_world_height - local_ground_world_height


def swing_foot_clearance_error(foot_clearance, foot_horizontal_speed):
    """Return a contact-free low-clearance cost active only for swing feet."""

    clearance = np.asarray(foot_clearance, dtype=np.float64)
    speed = np.asarray(foot_horizontal_speed, dtype=np.float64)
    swing_weight = np.clip(
        (speed - SWING_SPEED_START) / (SWING_SPEED_FULL - SWING_SPEED_START),
        0.0,
        1.0,
    )
    deficit = np.maximum(FOOT_CLEARANCE_TARGET - clearance, 0.0)
    return float(np.mean(swing_weight * np.square(deficit)))


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
    base_clearance: float,
    foot_clearance_error: float = 0.0,
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
    base_height_error = (float(base_clearance) - BASE_HEIGHT_TARGET) ** 2
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
        "foot_clearance": REWARD_DT
        * REWARD_SCALES["foot_clearance"]
        * float(foot_clearance_error),
        "action_rate": REWARD_DT
        * REWARD_SCALES["action_rate"]
        * action_rate,
        "similar_to_default": REWARD_DT
        * REWARD_SCALES["similar_to_default"]
        * similar_to_default,
    }
    return RewardTerms(**weighted, total=sum(weighted.values()))

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

from typing import Any


# Transfer checkpoint compatibility contract.
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
            "base_height_reference": "local_terrain_clearance",
            "foot_clearance_target": FOOT_CLEARANCE_TARGET,
            "foot_clearance_reference": "local_terrain_under_each_foot",
            "foot_swing_signal": "world_horizontal_foot_speed",
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
