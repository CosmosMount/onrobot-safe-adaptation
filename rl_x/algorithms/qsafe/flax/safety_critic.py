from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn


class SafetyQNetwork(nn.Module):
    observation_indices: Sequence[int]
    nr_hidden_units: int
    output_activation: str = "tanh"

    @nn.compact
    def __call__(self, observations, actions):
        observations = observations[..., jnp.asarray(self.observation_indices)]
        x = jnp.concatenate([observations, actions], axis=-1)
        x = nn.relu(nn.Dense(self.nr_hidden_units)(x))
        x = nn.relu(nn.Dense(self.nr_hidden_units)(x))
        x = nn.Dense(1)(x)
        if self.output_activation == "sigmoid":
            return nn.sigmoid(x)
        if self.output_activation == "tanh":
            return jnp.tanh(x)
        raise ValueError("QSafe output_activation must be tanh or sigmoid.")
