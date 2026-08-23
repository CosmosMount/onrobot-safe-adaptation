"""Backend-independent Go2 SQRL contracts."""

from .action import ActionMapper
from .observation import ObservationBuilder, build_observation
from .reward import compute_reward
from .specs import ACTION_SPEC, OBSERVATION_SPEC
from .types import RobotState, TrainingState

__all__ = [
    "ACTION_SPEC",
    "OBSERVATION_SPEC",
    "ActionMapper",
    "ObservationBuilder",
    "RobotState",
    "TrainingState",
    "build_observation",
    "compute_reward",
]

