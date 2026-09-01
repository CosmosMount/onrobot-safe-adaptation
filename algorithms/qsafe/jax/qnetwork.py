"""JAX/Flax QSafe network compatible with portable SQRL checkpoints."""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from algorithms.types import ObservationSpaceType


def get_q_network(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(
        env.general_properties,
        "critic_observation_indices",
        np.arange(env.general_properties.observation_space_shape[0]),
    )
    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return QNetwork(
            config.algorithm.nr_hidden_units,
            critic_observation_indices,
        )
    raise ValueError(
        f"unsupported QSafe observation space type {observation_space_type!r}"
    )


class QNetwork(nn.Module):
    nr_hidden_units: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observations, actions):
        observations = observations[..., self.critic_observation_indices]
        value = jnp.concatenate([observations, actions], axis=-1)
        value = nn.relu(nn.Dense(self.nr_hidden_units, name="Dense_0")(value))
        value = nn.relu(nn.Dense(self.nr_hidden_units, name="Dense_1")(value))
        value = nn.Dense(1, name="Dense_2")(value)
        return jnp.tanh(value)
