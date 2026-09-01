import unittest
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import torch

from algorithms.qsafe.jax.qnetwork import QNetwork as JaxQSafeNetwork
from algorithms.qsafe.pytorch.qnetwork import QNetwork as TorchQSafeNetwork
from algorithms.sac.jax.policy import Policy as JaxPolicy
from algorithms.sac.pytorch.policy import Policy as TorchPolicy
from algorithms.types import ActionSpaceType, ObservationSpaceType
from sqrl.checkpoint import flax_params, torch_state_to_arrays


class BoxStub:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.low = np.full(shape, -1.0, dtype=np.float32)
        self.high = np.full(shape, 1.0, dtype=np.float32)


class EnvStub:
    def __init__(self):
        self.single_action_space = BoxStub((2,))
        self.single_observation_space = BoxStub((4,))
        self.general_properties = SimpleNamespace(
            action_space_type=ActionSpaceType.CONTINUOUS,
            observation_space_type=ObservationSpaceType.FLAT_VALUES,
            observation_space_shape=(4,),
            policy_observation_indices=np.asarray([0, 1, 2, 3]),
            critic_observation_indices=np.asarray([0, 1, 2, 3]),
        )


class PortableCheckpointInteroperabilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.env = EnvStub()
        self.observations = np.asarray(
            [[0.2, -0.4, 0.7, 0.1], [-0.3, 0.8, -0.5, 0.4]],
            dtype=np.float32,
        )

    def test_policy_parameters_produce_the_same_jax_and_pytorch_outputs(self):
        torch_policy = TorchPolicy(
            self.env,
            log_std_min=-5.0,
            log_std_max=2.0,
            nr_hidden_units=3,
            device="cpu",
            policy_observation_indices=np.arange(4),
        )
        arrays = torch_state_to_arrays("policy", torch_policy.state_dict())
        parameters = flax_params(arrays, "policy")
        jax_policy = JaxPolicy((2,), -5.0, 2.0, 3, np.arange(4))

        with torch.no_grad():
            observations = torch.as_tensor(self.observations)
            latent = torch_policy.torso(observations)
            expected_mean = torch_policy.mean(latent).numpy()
            expected_log_std = torch_policy.log_std(latent).clamp(-5.0, 2.0).numpy()
        actual_mean, actual_log_std = jax_policy.apply(
            parameters, jnp.asarray(self.observations)
        )

        np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            actual_log_std, expected_log_std, rtol=1e-6, atol=1e-6
        )

    def test_qsafe_parameters_produce_the_same_jax_and_pytorch_outputs(self):
        torch_qsafe = TorchQSafeNetwork(
            self.env, 3, "cpu", np.arange(4)
        )
        arrays = torch_state_to_arrays("qsafe", torch_qsafe.state_dict())
        parameters = flax_params(arrays, "qsafe")
        jax_qsafe = JaxQSafeNetwork(3, np.arange(4))
        actions = np.asarray([[0.4, -0.2], [-0.1, 0.6]], dtype=np.float32)

        with torch.no_grad():
            expected = torch_qsafe(
                torch.as_tensor(self.observations), torch.as_tensor(actions)
            ).numpy()
        actual = jax_qsafe.apply(
            parameters,
            jnp.asarray(self.observations),
            jnp.asarray(actions),
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
