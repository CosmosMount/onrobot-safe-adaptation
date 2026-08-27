"""Shared sensor-free proprioceptive body-velocity estimation."""

from .velocity import VelocityEstimator
from .velocity_torch import TorchVelocityEstimator

__all__ = ["TorchVelocityEstimator", "VelocityEstimator"]
