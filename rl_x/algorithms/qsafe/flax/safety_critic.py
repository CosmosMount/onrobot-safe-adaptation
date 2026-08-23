from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn


class SafetyQNetwork(nn.Module):
    observation_indices: Sequence[int]
    nr_hidden_units: int

    @nn.compact
    def __call__(self, observations, actions):
        observations = observations[..., jnp.asarray(self.observation_indices)]
        x = jnp.concatenate([observations, actions], axis=-1)
        x = nn.relu(nn.Dense(self.nr_hidden_units)(x))
        x = nn.relu(nn.Dense(self.nr_hidden_units)(x))
        return jnp.tanh(nn.Dense(1)(x))
