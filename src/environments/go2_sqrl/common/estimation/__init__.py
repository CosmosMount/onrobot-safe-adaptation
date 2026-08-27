"""Shared sensor-free proprioceptive body-velocity estimation."""

from .velocity import (
    DEFAULT_VELOCITY_ESTIMATOR_CONFIG,
    VelocityEstimator,
    VelocityEstimatorConfig,
    configure_velocity_estimator,
    velocity_estimator_config_from,
)
from .velocity_torch import TorchVelocityEstimator

__all__ = [
    "DEFAULT_VELOCITY_ESTIMATOR_CONFIG",
    "TorchVelocityEstimator",
    "VelocityEstimator",
    "VelocityEstimatorConfig",
    "configure_velocity_estimator",
    "velocity_estimator_config_from",
]
