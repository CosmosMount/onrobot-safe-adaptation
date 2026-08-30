"""Shared Go2 policy, physics and environment contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch


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
DEFAULT_BASE_HEIGHT: Final = 0.27

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
class ObservationSpecV3:
    version: str = "go2-observation-v3-body-velocity"
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
class ActionSpecV2:
    version: str = "go2-action-v2-per-joint-scale"
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
class FailureSpecV3:
    """Sparse SQRL incident label shared by source and target backends."""

    version: str = "tilt-or-low-terrain-clearance-sustained-v3"
    angle_threshold: float = 0.8
    min_base_clearance: float = 0.18
    consecutive_frames: int = 5


# Canonical source/target tensor contract.
OBSERVATION_SPEC = ObservationSpecV3()
ACTION_SPEC = ActionSpecV2()
FAILURE_SPEC = FailureSpecV3()


def configure_failure_detection(config) -> None:
    """Expose the exact common SQRL failure label on an environment config."""

    config.fall_angle_threshold = FAILURE_SPEC.angle_threshold
    config.fall_min_base_clearance = FAILURE_SPEC.min_base_clearance
    config.fall_consecutive_frames = FAILURE_SPEC.consecutive_frames


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

from dataclasses import dataclass
from typing import Any

import numpy as np


# Backend-neutral state and transition records.
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
            "reward/action_rate": self.action_rate,
            "reward/similar_to_default": self.similar_to_default,
            "reward/total": self.total,
        }

import numpy as np


# Shared normalized-action projection.
class ActionMapper:
    def __init__(self, max_target_rate: float = ACTION_SPEC.max_target_rate):
        self.max_target_rate = float(max_target_rate)
        self.previous_q_target = DEFAULT_JOINT_POSITION.copy()

    def reset(self, previous_q_target: np.ndarray | None = None) -> None:
        self.previous_q_target = np.asarray(
            DEFAULT_JOINT_POSITION if previous_q_target is None else previous_q_target,
            dtype=np.float32,
        ).copy()

    def apply(self, action: np.ndarray) -> ActionResult:
        raw = np.asarray(action, dtype=np.float32)
        if raw.shape != (ACTION_SPEC.size,):
            raise ValueError(f"Action must have shape (12,), got {raw.shape}")
        applied, q_target = project_action_targets(
            self.previous_q_target,
            raw,
            max_target_rate=self.max_target_rate,
        )
        self.previous_q_target = q_target.copy()
        return ActionResult(raw.copy(), applied, q_target)


def _array_namespace(value):
    """Return NumPy or the optional Array API namespace of ``value``.

    JAX arrays expose ``__array_namespace__`` on supported versions.  Keeping
    the operations in that namespace makes the projection traceable without
    importing JAX in the SDK environment.
    """

    # JAX tracers do not consistently expose ``__array_namespace__`` across
    # versions, but their module remains under ``jax`` while tracing.
    if type(value).__module__.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return jnp
    namespace = getattr(value, "__array_namespace__", None)
    return namespace() if namespace is not None else np


def project_action_targets(
    previous_q_target,
    actions,
    *,
    max_target_rate: float = ACTION_SPEC.max_target_rate,
    array_namespace=None,
):
    """Project normalized actions without mutating environment state.

    Returns ``(applied_action, q_target)``.  The leading dimensions are
    arbitrary and broadcast normally; only the final dimension must be 12.
    """

    xp = array_namespace or _array_namespace(actions)
    if xp is np:
        action_array = xp.asarray(actions, dtype=np.float32)
        previous_array = xp.asarray(previous_q_target, dtype=np.float32)
    else:
        action_array = xp.asarray(actions)
        previous_array = xp.asarray(previous_q_target)
    if action_array.shape[-1] != ACTION_SPEC.size:
        raise ValueError(f"Action must end in 12 values, got {action_array.shape}")
    if previous_array.shape[-1] != ACTION_SPEC.size:
        raise ValueError(
            "previous_q_target must end in 12 values, "
            f"got {previous_array.shape}"
        )

    dtype = action_array.dtype
    default = xp.asarray(DEFAULT_JOINT_POSITION, dtype=dtype)
    lower = xp.asarray(JOINT_LOWER_LIMIT, dtype=dtype)
    upper = xp.asarray(JOINT_UPPER_LIMIT, dtype=dtype)
    scale = xp.asarray(ACTION_SPEC.scale, dtype=dtype)
    clipped = xp.clip(action_array, -1.0, 1.0)
    q_target = xp.clip(default + scale * clipped, lower, upper)
    max_delta = float(max_target_rate) * ACTION_SPEC.control_dt
    q_target = xp.clip(
        q_target,
        previous_array - max_delta,
        previous_array + max_delta,
    )
    applied = xp.clip((q_target - default) / scale, -1.0, 1.0)
    return applied, q_target


def project_actions_from_observation(
    observations,
    actions,
    *,
    max_target_rate: float = ACTION_SPEC.max_target_rate,
):
    """Return applied normalized actions using the observation's prior target."""

    xp = _array_namespace(actions)
    observation_array = xp.asarray(observations)
    if observation_array.shape[-1] != OBSERVATION_SPEC.size:
        raise ValueError(
            f"Observation must end in {OBSERVATION_SPEC.size} values, "
            f"got {observation_array.shape}"
        )
    applied, _ = project_action_targets(
        observation_array[..., OBSERVATION_SPEC.previous_action_q_target],
        actions,
        max_target_rate=max_target_rate,
        array_namespace=xp,
    )
    return applied


def project_action_targets_tensor(previous_q_target, actions):
    """Differentiable Torch counterpart of the shared action projection."""

    if not torch.is_tensor(actions):
        actions = torch.as_tensor(actions, dtype=torch.float32)
    if not torch.is_tensor(previous_q_target):
        previous_q_target = torch.as_tensor(previous_q_target, dtype=actions.dtype, device=actions.device)
    else:
        previous_q_target = previous_q_target.to(dtype=actions.dtype, device=actions.device)
    default = torch.as_tensor(DEFAULT_JOINT_POSITION, dtype=actions.dtype, device=actions.device)
    scale = torch.as_tensor(ACTION_SPEC.scale, dtype=actions.dtype, device=actions.device)
    lower = torch.as_tensor(JOINT_LOWER_LIMIT, dtype=actions.dtype, device=actions.device)
    upper = torch.as_tensor(JOINT_UPPER_LIMIT, dtype=actions.dtype, device=actions.device)
    target = (default + scale * actions.clamp(-1, 1)).clamp(lower, upper)
    max_delta = ACTION_SPEC.max_target_rate * ACTION_SPEC.control_dt
    target = torch.maximum(torch.minimum(target, previous_q_target + max_delta), previous_q_target - max_delta)
    return ((target - default) / scale).clamp(-1, 1), target

import numpy as np

# Common vector-environment interface implemented by both backends.
class Go2Environment:
    """Shared vector-environment contract for the Isaac and MuJoCo backends."""

    def __init__(self, config, nr_envs):
        from types import SimpleNamespace
        from core.types import ActionSpaceType, ObservationSpaceType
        from gymnasium.spaces import Box

        self.config = config.environment
        self.nr_envs = self.num_envs = int(nr_envs)
        self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.single_action_space = Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.policy_observation_indices = np.arange(OBSERVATION_SIZE)
        self.critic_observation_indices = np.arange(OBSERVATION_SIZE)
        self.safety_critic_observation_indices = np.arange(OBSERVATION_SIZE)
        self.general_properties = SimpleNamespace(
            action_space_type=ActionSpaceType.CONTINUOUS,
            observation_space_type=ObservationSpaceType.FLAT_VALUES,
            observation_space_shape=self.single_observation_space.shape,
            policy_observation_indices=self.policy_observation_indices,
        )

    @staticmethod
    def project_actions(states, actions):
        if torch.is_tensor(actions):
            if not torch.is_tensor(states):
                states = torch.as_tensor(states, dtype=actions.dtype, device=actions.device)
            else:
                states = states.to(dtype=actions.dtype, device=actions.device)
            if states.shape[-1] != OBSERVATION_SIZE:
                raise ValueError(f"Observation must end in {OBSERVATION_SIZE} values, got {states.shape}")
            return project_action_targets_tensor(states[..., OBSERVATION_SPEC.previous_action_q_target], actions)[0]
        return project_actions_from_observation(states, actions)

    @staticmethod
    def get_final_observation_at_index(info, index):
        return info["final_observation"][index]

    @staticmethod
    def get_final_info_value_at_index(info, key, index):
        final_info = info["final_info"][index]
        if final_info is None:
            raise KeyError(f"No final info for environment {index}")
        return final_info[key]

    @staticmethod
    def get_logging_info_dict(info):
        ignored = {"failure", "applied_action", "final_observation", "final_info"}
        return {key: np.asarray(value).reshape(-1).tolist() for key, value in info.items()
                if key not in ignored and not isinstance(value, list)}

    def checkpoint_manifest(self, normalizer=None):
        from .task import build_manifest

        return build_manifest(
            normalizer,
            fall_angle_threshold=float(self.config.fall_angle_threshold),
            fall_min_base_clearance=float(self.config.fall_min_base_clearance),
            fall_consecutive_frames=int(self.config.fall_consecutive_frames),
            target_velocity_x=float(self.config.target_velocity_x),
        )

    def validate_checkpoint_manifest(self, manifest, normalizer=None):
        from .task import validate_manifest

        validate_manifest(manifest, self.checkpoint_manifest(normalizer))

    def validate_transfer_checkpoint_manifest(self, manifest, normalizer=None):
        from .task import validate_transfer_manifest

        validate_transfer_manifest(manifest, self.checkpoint_manifest(normalizer))
