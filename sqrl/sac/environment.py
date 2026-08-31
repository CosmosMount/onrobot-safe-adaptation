from dataclasses import dataclass
from typing import Any

import numpy as np


class InvalidTransitionError(RuntimeError):
    """Infrastructure failure whose transition must not enter replay."""


@dataclass(slots=True)
class RolloutStep:
    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: dict[str, Any]


def resolve_executed_actions(info, proposed_actions, nr_envs, action_shape):
    """Return the normalized action actually applied by the environment."""

    source = (
        info["applied_action"]
        if isinstance(info, dict) and "applied_action" in info
        else proposed_actions
    )
    actions = np.asarray(source, dtype=np.float32)
    expected_shape = (int(nr_envs),) + tuple(action_shape)
    if actions.shape != expected_shape:
        raise ValueError(
            f"Executed actions must have shape {expected_shape}, got {actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("Executed actions must contain only finite values")
    return actions
