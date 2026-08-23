"""Optional generic contracts for safe-RL environment interaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class InvalidTransitionError(RuntimeError):
    """An infrastructure error whose transition must not enter replay."""


@dataclass(slots=True)
class RolloutStep:
    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: dict[str, Any]


@runtime_checkable
class PartitionedSafetyRolloutEnvironment(Protocol):
    """A single simulator containing isolated task and safety pools."""

    nr_task_envs: int
    nr_safety_envs: int

    def reset_partitions(self) -> tuple[Any, Any]: ...

    def step_partitions(
        self, task_actions: Any, safety_actions: Any
    ) -> tuple[RolloutStep, RolloutStep]: ...

