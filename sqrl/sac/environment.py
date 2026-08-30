from dataclasses import dataclass
from typing import Any


class InvalidTransitionError(RuntimeError):
    """Infrastructure failure whose transition must not enter replay."""


@dataclass(slots=True)
class RolloutStep:
    observation: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: dict[str, Any]
