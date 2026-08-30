"""Shared Go2 observation, reward, termination, and checkpoint contracts."""
from __future__ import annotations

from typing import Protocol, Sequence

from .base import (
    ACTION_SPEC, CONTACT_FRICTION, CONTROL_DT, DEFAULT_BASE_HEIGHT,
    DEFAULT_JOINT_POSITION, EPISODE_STEPS, FAILURE_SPEC, GRAVITY_Z,
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


REWARD_SCALES = {
    "tracking_velocity": 1.0,
    "yaw_rate": -0.1,
    "upright": -10.0,
    "energy": -0.0003,
}


def local_base_clearance(base_world_height, local_ground_world_height):
    """Return base height measured from the local terrain surface."""

    return base_world_height - local_ground_world_height


def quaternion_to_rpy_wxyz(
    quaternion: np.ndarray,
) -> tuple[float, float, float]:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


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

from typing import Any


def build_manifest(
    normalizer: dict[str, Any] | None = None,
    *,
    fall_angle_threshold: float = FAILURE_SPEC.angle_threshold,
    fall_min_base_clearance: float = FAILURE_SPEC.min_base_clearance,
    fall_consecutive_frames: int = FAILURE_SPEC.consecutive_frames,
    target_velocity_x: float = 0.5,
    domain_randomization: bool = False,
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
