import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def squashed_gaussian_log_probability(pretanh, mean, log_std):
    """Return ``log pi(tanh(z) | state)`` over action dimensions.

    Computing the tanh Jacobian as ``log(1 - tanh(z) ** 2)`` loses both
    numerical precision and its useful gradient once ``tanh(z)`` rounds to
    ``+/-1``.  This softplus identity is algebraically equivalent but remains
    finite and differentiable for large pre-tanh samples.
    """

    std = torch.exp(log_std)
    gaussian_log_probability = (
        -0.5 * ((pretanh - mean) / std).pow(2)
        - 0.5 * math.log(2.0 * math.pi)
        - log_std
    )
    tanh_log_abs_det_jacobian = 2.0 * (
        math.log(2.0) - pretanh - F.softplus(-2.0 * pretanh)
    )
    return (gaussian_log_probability - tanh_log_abs_det_jacobian).sum(dim=-1)


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))
    compile_mode = config.algorithm.compile_mode

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = Policy(env, config.algorithm.log_std_min, config.algorithm.log_std_max, config.algorithm.nr_hidden_units, device, policy_observation_indices).to(device)
        if not bool(config.algorithm.compile_policy):
            return policy
        policy = torch.compile(policy, mode=compile_mode)
        policy.forward = torch.compile(policy.forward, mode=compile_mode)
        policy.get_action = torch.compile(policy.get_action, mode=compile_mode)
        policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=compile_mode)
        return policy
    

class Policy(nn.Module):
    def __init__(self, env, log_std_min, log_std_max, nr_hidden_units, device, policy_observation_indices):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32).to(device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32).to(device)
        single_as_shape = env.single_action_space.shape
        obs_input_dim = len(policy_observation_indices)

        self.torso = nn.Sequential(
            nn.Linear(obs_input_dim, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
        )
        self.mean = nn.Linear(nr_hidden_units, np.prod(single_as_shape, dtype=int).item())
        self.log_std = nn.Linear(nr_hidden_units, np.prod(single_as_shape, dtype=int).item())


    def get_action(self, x):
        x = x[..., self.policy_observation_indices]
        latent = self.torso(x)
        mean = self.mean(latent)
        log_std = self.log_std(latent)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        normal = Normal(mean, std)
        action = normal.rsample()  # Reparameterization trick
        action_tanh = torch.tanh(action)

        log_prob = squashed_gaussian_log_probability(action, mean, log_std)
        log_prob = log_prob.unsqueeze(-1)

        scaled_action = self.env_as_low + (0.5 * (action_tanh + 1.0) * (self.env_as_high - self.env_as_low))

        return action_tanh, scaled_action, log_prob


    def get_deterministic_action(self, x):
        x = x[..., self.policy_observation_indices]
        latent = self.torso(x)
        mean = self.mean(latent)
        action_tanh = torch.tanh(mean)
        scaled_action = self.env_as_low + (0.5 * (action_tanh + 1.0) * (self.env_as_high - self.env_as_low))
        return scaled_action
