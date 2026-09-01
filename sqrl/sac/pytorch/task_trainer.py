"""Task SAC trainer used by SQRL pre-training."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast

from algorithms.sac.pytorch.critic import get_critic as get_task_critic
from algorithms.sac.pytorch.entropy_coefficient import get_entropy_coefficient
from algorithms.sac.pytorch.policy import get_policy
from sqrl.sac.pytorch.replay_buffer import (
    ReplayBuffer,
    newly_eligible_transitions,
    validate_policy_commands,
)


class TaskTrainer:
    """Train the SAC task policy on one standard vector environment."""

    def __init__(self, config, env, device, rng):
        self.env = env
        self.device = torch.device(device)
        algorithm = config.algorithm

        self.nr_envs = int(env.num_envs)
        self.bf16 = bool(algorithm.bf16_mixed_precision_training)
        self.gamma = float(algorithm.gamma)
        self.tau = float(algorithm.tau)
        self.batch_size = int(algorithm.batch_size)
        self.learning_starts = int(algorithm.get("learning_starts", 0))
        self.utd_ratio = float(algorithm.get("task_utd_ratio", 1.0))

        if self.learning_starts < 0:
            raise ValueError("learning_starts must be non-negative")
        if not np.isfinite(self.utd_ratio) or self.utd_ratio < 0.0:
            raise ValueError("task_utd_ratio must be a finite non-negative value")
        if self.bf16 and self.device.type != "cuda":
            raise ValueError("bfloat16 mixed precision training requires CUDA")

        self.policy = get_policy(config, env, self.device)
        self.critic = get_task_critic(config, env, self.device)
        self.entropy = get_entropy_coefficient(config, env, self.device)

        learning_rate = float(algorithm.get("learning_rate", 3e-4))
        optimizer_options = {"fused": True} if self.device.type == "cuda" else {}
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=float(algorithm.get("policy_lr", learning_rate)),
            **optimizer_options,
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()),
            lr=float(algorithm.get("qtask_lr", learning_rate)),
            **optimizer_options,
        )
        self.entropy_optimizer = optim.Adam(
            [self.entropy.log_alpha],
            lr=float(algorithm.get("entropy_lr", learning_rate)),
            **optimizer_options,
        )

        buffer_size = int(
            algorithm.get(
                "task_buffer_size", algorithm.get("buffer_size", 1_000_000)
            )
        )
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            nr_envs=self.nr_envs,
            os_shape=env.single_observation_space.shape,
            as_shape=env.single_action_space.shape,
            rng=rng,
            device=self.device,
        )
        self.action_low = np.asarray(env.single_action_space.low, dtype=np.float32)
        self.action_high = np.asarray(env.single_action_space.high, dtype=np.float32)

        self.state = None
        self.steps = 0
        self.updates = 0
        self.update_credit = 0.0

    def set_state(self, state):
        """Set the current task observation after the initial reset."""

        self.state = state

    def train_step(self):
        """Collect one vector step and perform the eligible SAC updates."""

        self.policy.train()
        self.critic.q1.train()
        self.critic.q2.train()

        actions, processed_actions = self._sample_actions()
        actions = validate_policy_commands(
            actions, self.nr_envs, self.env.single_action_space.shape
        )
        next_state, rewards, terminated, truncated, info = self.env.step(
            processed_actions
        )
        actual_next_state, terminations, failures = self._process_step(
            next_state, terminated, truncated, info
        )
        # Replay the policy command; actuator projection is part of env.step.
        self.replay_buffer.add(
            np.asarray(self.state, dtype=np.float32),
            actual_next_state,
            actions,
            np.asarray(rewards, dtype=np.float32),
            terminations.astype(np.float32),
            failures,
        )
        self.state = next_state

        previous_steps = self.steps
        self.steps += self.nr_envs
        self.update_credit += newly_eligible_transitions(
            previous_steps,
            self.steps,
            self.learning_starts,
        ) * self.utd_ratio
        nr_updates = int(np.floor(self.update_credit + 1e-12))
        self.update_credit -= nr_updates

        metrics = None
        for _ in range(nr_updates):
            metrics = self._update_networks()
        return metrics

    def _sample_actions(self):
        """Sample one vector of task-policy commands."""

        if self.steps < self.learning_starts:
            processed_actions = np.asarray(
                [self.env.single_action_space.sample() for _ in range(self.nr_envs)],
                dtype=np.float32,
            )
            actions = 2.0 * (processed_actions - self.action_low) / (
                self.action_high - self.action_low
            ) - 1.0
            return actions, processed_actions

        with torch.no_grad(), autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16
        ):
            actions, processed_actions, _ = self.policy.get_action(
                torch.as_tensor(self.state, dtype=torch.float32, device=self.device)
            )
        return actions.cpu().numpy(), processed_actions.cpu().numpy()

    def _process_step(self, next_state, terminated, truncated, info):
        terminations = np.asarray(terminated, dtype=bool).reshape(self.nr_envs)
        truncations = np.asarray(truncated, dtype=bool).reshape(self.nr_envs)
        actual_next_state = np.asarray(next_state, dtype=np.float32).copy()
        for index in np.flatnonzero(terminations | truncations):
            actual_next_state[index] = np.asarray(
                self.env.get_final_observation_at_index(info, index),
                dtype=np.float32,
            )

        if isinstance(info, dict) and "failure" in info:
            failures = np.asarray(info["failure"], dtype=np.float32)
        elif isinstance(info, dict) and "failures" in info:
            failures = np.asarray(info["failures"], dtype=np.float32)
        else:
            failures = np.zeros(self.nr_envs, dtype=np.float32)
        return actual_next_state, terminations, failures

    def _update_networks(self):
        states, next_states, actions, rewards, terminations, _ = (
            self.replay_buffer.sample(self.batch_size)
        )
        with autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16
        ):
            with torch.no_grad():
                next_actions, _, next_log_probs = self.policy.get_action(next_states)
                next_q = torch.minimum(
                    self.critic.q1_target(next_states, next_actions),
                    self.critic.q2_target(next_states, next_actions),
                )
                target_q = rewards.reshape(-1, 1) + self.gamma * (
                    1.0 - terminations.reshape(-1, 1)
                ) * (next_q - self.entropy().detach() * next_log_probs)
            q1 = self.critic.q1(states, actions)
            q2 = self.critic.q2(states, actions)
            critic_loss = 0.5 * (
                F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self._update_targets()

        with autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16
        ):
            current_actions, _, current_log_probs = self.policy.get_action(states)
            current_q = torch.minimum(
                self.critic.q1(states, current_actions),
                self.critic.q2(states, current_actions),
            )
            policy_loss = (
                self.entropy().detach() * current_log_probs - current_q
            ).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        entropy_loss = self.entropy.loss(-current_log_probs.detach()).mean()
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()
        self.updates += 1
        return {
            "qtask_loss": critic_loss.detach().item(),
            "policy_loss": policy_loss.detach().item(),
            "entropy_loss": entropy_loss.detach().item(),
        }

    def _update_targets(self):
        with torch.no_grad():
            for online, target in zip(
                self.critic.q1.parameters(), self.critic.q1_target.parameters()
            ):
                target.data.mul_(1.0 - self.tau).add_(online.data, alpha=self.tau)
            for online, target in zip(
                self.critic.q2.parameters(), self.critic.q2_target.parameters()
            ):
                target.data.mul_(1.0 - self.tau).add_(online.data, alpha=self.tau)
