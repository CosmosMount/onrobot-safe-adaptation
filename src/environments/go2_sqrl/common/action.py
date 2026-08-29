"""Shared Go2 normalized-action to joint-target pipeline."""

from __future__ import annotations

import numpy as np

from .specs import (
    ACTION_SPEC,
    DEFAULT_JOINT_POSITION,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    OBSERVATION_SPEC,
)
from .types import ActionResult


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


def normalized_action_from_target(q_target: np.ndarray) -> np.ndarray:
    q_target = np.asarray(q_target, dtype=np.float32)
    scale = np.asarray(ACTION_SPEC.scale, dtype=np.float32)
    return np.clip(
        (q_target - DEFAULT_JOINT_POSITION) / scale, -1.0, 1.0
    ).astype(np.float32)
