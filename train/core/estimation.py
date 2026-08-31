"""Public backend-independent velocity-estimation API."""

from .estimation_numpy import (
    DEFAULT_VELOCITY_ESTIMATOR_CONFIG,
    VelocityEstimator,
    VelocityEstimatorConfig,
    configure_velocity_estimator,
    foot_position_velocity_body,
    quaternion_rotation_matrix_wxyz,
    velocity_estimator_config_from,
)
from .estimation_torch import (
    TorchVelocityEstimator,
    quaternion_rotation_matrix_wxyz_torch,
)

__all__ = [
    "DEFAULT_VELOCITY_ESTIMATOR_CONFIG",
    "TorchVelocityEstimator",
    "VelocityEstimator",
    "VelocityEstimatorConfig",
    "configure_velocity_estimator",
    "foot_position_velocity_body",
    "quaternion_rotation_matrix_wxyz",
    "quaternion_rotation_matrix_wxyz_torch",
    "velocity_estimator_config_from",
]

