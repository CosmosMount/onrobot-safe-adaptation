import numpy as np
import torch
import torch.nn as nn


class SafetyQNetwork(nn.Module):
    """SQRL failure-probability critic, independent from the task critic."""

    def __init__(
        self,
        observation_shape,
        action_shape,
        observation_indices,
        hidden_units,
        output_activation="tanh",
    ):
        super().__init__()
        indices = np.asarray(observation_indices, dtype=np.int64)
        self.register_buffer(
            "observation_indices", torch.as_tensor(indices, dtype=torch.long)
        )
        input_dim = len(indices) + int(np.prod(action_shape))
        activation = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid}.get(
            str(output_activation)
        )
        if activation is None:
            raise ValueError("QSafe output_activation must be tanh or sigmoid.")
        self.output_activation = str(output_activation)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
            activation(),
        )

    def forward(self, observations, actions):
        observations = observations[..., self.observation_indices]
        return self.network(torch.cat([observations, actions], dim=-1))
