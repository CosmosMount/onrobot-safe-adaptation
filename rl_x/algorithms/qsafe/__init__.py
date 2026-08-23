"""Task-critic-independent safety Q-function components."""

from rl_x.algorithms.qsafe.common import extract_failure_signal
from rl_x.algorithms.qsafe.interface import QSafeComponent

__all__ = ["extract_failure_signal", "QSafeComponent"]
