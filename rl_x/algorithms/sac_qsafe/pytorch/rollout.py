"""Rollout bookkeeping independent of any concrete simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def preserve_policy_outputs(*outputs):
    """Clone compiled-policy outputs that must survive another graph replay.

    ``torch.compile(mode="reduce-overhead")`` may reuse CUDA Graph output
    buffers on the next invocation.  A clone made by the Python caller owns
    independent storage and is therefore safe to consume afterwards.
    """

    return tuple(output.clone() for output in outputs)


class TransitionUpdateBudget:
    """Accumulate optimizer-update credit per newly collected transition.

    ``update_ratio`` is deliberately defined as *gradient updates per new
    transition*, rather than per vector-environment call.  This keeps the
    optimization budget invariant when the number of parallel environments
    changes.  Fractional ratios are retained as credit for later calls.
    """

    def __init__(self, update_ratio: float):
        update_ratio = float(update_ratio)
        if not math.isfinite(update_ratio) or update_ratio < 0.0:
            raise ValueError("update_ratio must be a finite non-negative value")
        self.update_ratio = update_ratio
        self.credit = 0.0
        self.transitions = 0
        self.updates = 0

    def add_transitions(self, nr_transitions: int) -> None:
        nr_transitions = int(nr_transitions)
        if nr_transitions < 0:
            raise ValueError("nr_transitions must be non-negative")
        self.transitions += nr_transitions
        self.credit += nr_transitions * self.update_ratio

    def consume_ready_updates(self) -> int:
        # The small tolerance prevents values such as 0.9999999999999999 from
        # postponing an update indefinitely when a decimal ratio is used.
        nr_updates = int(math.floor(self.credit + 1e-12))
        self.credit -= nr_updates
        if self.credit < 0.0 and self.credit > -1e-9:
            self.credit = 0.0
        self.updates += nr_updates
        return nr_updates

    @property
    def effective_ratio(self) -> float:
        if self.transitions == 0:
            return 0.0
        return self.updates / self.transitions


def completed_trajectory_transition_count(completed_trajectories) -> int:
    """Count transitions committed atomically as complete trajectories."""

    return sum(len(trajectory) for trajectory in completed_trajectories)


class AtomicTrajectoryUpdateBudget:
    """Issue update credit only when complete trajectories are committed.

    Safety trajectories may have different lengths, so using one update per
    transition can create an enormous synchronized burst at time limits.  The
    configured ratio therefore means updates per *completed trajectory*, while
    transition and trajectory totals are both retained for observability.
    """

    def __init__(self, updates_per_trajectory: float):
        self._credit_budget = TransitionUpdateBudget(updates_per_trajectory)
        self.transitions = 0
        self.trajectories = 0

    @property
    def update_ratio(self) -> float:
        return self._credit_budget.update_ratio

    @property
    def credit(self) -> float:
        return self._credit_budget.credit

    @property
    def updates(self) -> int:
        return self._credit_budget.updates

    @property
    def effective_ratio(self) -> float:
        if self.trajectories == 0:
            return 0.0
        return self.updates / self.trajectories

    def add_completed(self, completed_trajectories) -> int:
        completed_trajectories = tuple(completed_trajectories)
        nr_transitions = completed_trajectory_transition_count(
            completed_trajectories
        )
        self.transitions += nr_transitions
        self.trajectories += len(completed_trajectories)
        self._credit_budget.add_transitions(len(completed_trajectories))
        return nr_transitions

    def consume_ready_updates(self) -> int:
        return self._credit_budget.consume_ready_updates()


@dataclass(slots=True)
class PartitionState:
    task_observation: Any
    safety_observation: Any
    task_transitions: int = 0
    safety_transitions: int = 0


class PartitionedRolloutCounter:
    """Count task and safety interaction streams without mixing budgets."""

    def __init__(self, nr_task_envs: int, nr_safety_envs: int):
        if nr_task_envs < 1 or nr_safety_envs < 1:
            raise ValueError("Both task and safety pools must be non-empty")
        self.nr_task_envs = int(nr_task_envs)
        self.nr_safety_envs = int(nr_safety_envs)
        self.task_transitions = 0
        self.safety_transitions = 0

    def advance(self) -> tuple[int, int]:
        self.task_transitions += self.nr_task_envs
        self.safety_transitions += self.nr_safety_envs
        return self.task_transitions, self.safety_transitions
