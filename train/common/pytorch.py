"""Shared PyTorch policy construction and checkpoint handling."""
import os

import numpy as np
import torch
import torch.nn as nn

from algorithms.qsafe.pytorch.critic import get_critic as get_safe_critic
from algorithms.sac.pytorch.policy import get_policy
from train.common.base import OBSERVATION_SPEC, project_action_targets_tensor


class ProjectedPolicy(nn.Module):
    """Apply the shared Go2 executable-action contract to every policy action."""

    def __init__(self, policy, env, device):
        super().__init__()
        self.policy = policy
        self.previous_target_slice = OBSERVATION_SPEC.previous_action_q_target
        self.register_buffer("env_low", torch.as_tensor(env.single_action_space.low, dtype=torch.float32, device=device))
        self.register_buffer("env_high", torch.as_tensor(env.single_action_space.high, dtype=torch.float32, device=device))

    def project(self, states, actions):
        return project_action_targets_tensor(states[..., self.previous_target_slice], actions)[0]

    def get_action(self, states):
        actions, _, log_probs = self.policy.get_action(states)
        actions = self.project(states, actions)
        processed_actions = self.env_low + 0.5 * (actions + 1.0) * (self.env_high - self.env_low)
        return actions, processed_actions, log_probs

    def get_deterministic_action(self, states):
        processed_actions = self.policy.get_deterministic_action(states)
        actions = 2.0 * (processed_actions - self.env_low) / (self.env_high - self.env_low) - 1.0
        actions = self.project(states, actions)
        return self.env_low + 0.5 * (actions + 1.0) * (self.env_high - self.env_low)


class SQRLTrainingBase:
    def __init__(self, config, env, device):
        self.config = config
        self.env = env
        self.device = torch.device(device)
        self.rng = np.random.default_rng(int(config.environment.seed))
        torch.manual_seed(int(config.environment.seed))
        torch.backends.cudnn.deterministic = True
        base_policy = get_policy(config, env, self.device)
        self.policy = ProjectedPolicy(base_policy, env, self.device)
        self.safe_critic = get_safe_critic(config, env, self.device)

    def save(self, path, phase):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        manifest = self.env.checkpoint_manifest(None) if hasattr(self.env, "checkpoint_manifest") else None
        torch.save(
            {
                "phase": str(phase),
                "policy": self.policy.state_dict(),
                "qsafe": self.safe_critic.q.state_dict(),
                "manifest": manifest,
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config),
            },
            path,
        )

    def load(self, path, transfer=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        manifest = checkpoint.get("manifest")
        validator_name = "validate_transfer_checkpoint_manifest" if transfer else "validate_checkpoint_manifest"
        validator = getattr(self.env, validator_name, None)
        if manifest is not None and validator is not None:
            validator(manifest, None)
        self.policy.load_state_dict(checkpoint["policy"])
        self.safe_critic.q.load_state_dict(checkpoint["qsafe"])
        self.safe_critic.q_target.load_state_dict(checkpoint["qsafe"])
        return checkpoint.get("phase")
