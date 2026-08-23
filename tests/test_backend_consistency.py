import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.environments.go2_sqrl.common.observation import build_observation
from src.environments.go2_sqrl.common.specs import DEFAULT_JOINT_POSITION
from src.environments.go2_sqrl.common.types import RobotState
from src.environments.go2_sqrl.isaac_lab.mdp import build_observation_tensor
from src.environments.go2_sqrl.common.estimation.velocity_torch import (
    TorchVelocityEstimator,
)


def test_numpy_and_isaac_torch_observation_adapters_match():
    joint_q = DEFAULT_JOINT_POSITION + np.linspace(-0.1, 0.1, 12, dtype=np.float32)
    joint_dq = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    gyro = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    velocity = np.asarray([0.4, -0.1, 0.0], dtype=np.float32)
    quaternion = np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    previous_target = DEFAULT_JOINT_POSITION + 0.01
    state = RobotState(joint_q, joint_dq, gyro, quaternion)
    numpy_observation, _ = build_observation(
        state, velocity, previous_target, previous_quaternion=None
    )
    torch_observation, _ = build_observation_tensor(
        torch.tensor(joint_q)[None],
        torch.tensor(joint_dq)[None],
        torch.tensor(gyro)[None],
        torch.tensor(velocity)[None],
        torch.tensor(quaternion)[None],
        torch.tensor(previous_target)[None],
    )
    np.testing.assert_allclose(
        numpy_observation, torch_observation[0].numpy(), atol=1e-6
    )


def test_torch_velocity_estimator_is_stationary_at_nominal_stance():
    estimator = TorchVelocityEstimator(1, "cpu")
    velocity = estimator.update(
        torch.tensor(DEFAULT_JOINT_POSITION)[None],
        torch.zeros((1, 12)),
        torch.zeros((1, 3)),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 9.81]]),
    )
    torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=1e-6, rtol=0)
