"""Shared episode and invalid-transition semantics."""

from __future__ import annotations


class InvalidTransition(RuntimeError):
    """Infrastructure failure whose transition must not enter replay."""


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

