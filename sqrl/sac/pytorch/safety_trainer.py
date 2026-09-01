"""Safety rollout trainer used by SQRL pre-training."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast

from algorithms.qsafe.pytorch.critic import get_critic as get_safe_critic
from sqrl.sac.pytorch.replay_buffer import validate_policy_commands
from sqrl.sac.pytorch.rollout_buffer import RolloutBuffer
from sqrl.sac.pytorch.safety_ops import (
    qsafe_bellman_target,
    qsafe_optimizer_steps_for_block,
    sample_safe_actions,
)


def qsafe_update_schedule(algorithm) -> tuple[int | None, float | None]:
    """Return the paper-default single update or an explicit epoch extension."""

    steps_value = algorithm.get("qsafe_optimizer_steps_per_block", None)
    epochs_value = algorithm.get("qsafe_epochs_per_block", None)
    if steps_value is not None and epochs_value is not None:
        raise ValueError(
            "Configure only one of qsafe_optimizer_steps_per_block and "
            "qsafe_epochs_per_block"
        )
    if epochs_value is not None:
        epochs = float(epochs_value)
        if not np.isfinite(epochs) or epochs <= 0.0:
            raise ValueError("qsafe_epochs_per_block must be finite and positive")
        return None, epochs

    steps_value = 1 if steps_value is None else steps_value
    steps = int(steps_value)
    if isinstance(steps_value, bool) or float(steps_value) != steps or steps < 1:
        raise ValueError("qsafe_optimizer_steps_per_block must be a positive integer")
    return steps, None


class SafetyTrainer:
    """Collect on-policy safety rollouts and update only QSafe."""

    def __init__(self, config, env, policy, device, rng):
        self.env = env
        self.policy = policy
        self.device = torch.device(device)
        algorithm = config.algorithm

        self.nr_envs = int(env.num_envs)
        self.bf16 = bool(algorithm.bf16_mixed_precision_training)
        self.gamma = float(algorithm.get("safe_gamma", 0.7))
        self.tau = float(algorithm.get("safe_tau", algorithm.tau))
        self.batch_size = int(
            algorithm.get("safety_batch_size", algorithm.batch_size)
        )
        self.rollouts_per_block = int(algorithm.k)
        self.epsilon = float(algorithm.get("epsilon_safe", 0.1))
        self.nr_action_samples = int(
            algorithm.get("max_safe_action_samples", 100)
        )
        self.optimizer_steps, self.epochs_per_block = qsafe_update_schedule(
            algorithm
        )

        if self.rollouts_per_block < 1:
            raise ValueError("k must be at least 1")
        if self.nr_action_samples < 1:
            raise ValueError("max_safe_action_samples must be at least 1")
        if self.bf16 and self.device.type != "cuda":
            raise ValueError("bfloat16 mixed precision training requires CUDA")

        self.critic = get_safe_critic(config, env, self.device)
        learning_rate = float(algorithm.get("learning_rate", 3e-4))
        optimizer_options = {"fused": True} if self.device.type == "cuda" else {}
        self.optimizer = optim.Adam(
            self.critic.q.parameters(),
            lr=float(algorithm.get("qsafe_lr", learning_rate)),
            **optimizer_options,
        )

        buffer_size = int(algorithm.get("safety_buffer_size", 100_000))
        max_trajectories = int(
            algorithm.get("max_safety_trajectories", self.rollouts_per_block)
        )
        self.replay_buffer = RolloutBuffer(
            capacity=buffer_size,
            nr_envs=self.nr_envs,
            os_shape=env.single_observation_space.shape,
            as_shape=env.single_action_space.shape,
            rng=rng,
            max_trajectories=max_trajectories,
        )
        self.rollouts = 0
        self.blocks = 0
        self.update_steps = 0

    def train_block(self):
        """Collect ``k`` complete safety rollouts, then update QSafe."""

        self.policy.eval()
        self.critic.q.train()
        reset_result = self.env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        collected = 0
        selected_q_sum = 0.0
        fallback_count = 0
        action_count = 0
        while collected < self.rollouts_per_block:
            actions, processed_actions, selected_q, fallback = self._sample_actions(
                state
            )
            actions = validate_policy_commands(
                actions, self.nr_envs, self.env.single_action_space.shape
            )
            selected_q_sum += float(selected_q.sum())
            fallback_count += int(fallback.sum())
            action_count += int(fallback.size)
            next_state, _, terminated, truncated, info = self.env.step(
                processed_actions
            )
            actual_next_state, terminations, truncations, failures = (
                self._process_step(next_state, terminated, truncated, info)
            )
            # Replay the policy command; actuator projection is part of env.step.
            completed = self.replay_buffer.add(
                np.asarray(state, dtype=np.float32),
                actual_next_state,
                actions,
                failures,
                terminations,
                truncations,
                self.rollouts_per_block - collected,
            )
            collected += completed
            self.rollouts += completed
            state = next_state

        nr_updates = self.optimizer_steps
        if nr_updates is None:
            nr_updates = qsafe_optimizer_steps_for_block(
                self.replay_buffer.size,
                self.batch_size,
                self.epochs_per_block,
            )
        updates = [self._update_critic() for _ in range(nr_updates)]
        metrics = {
            key: float(np.mean([update[key] for update in updates]))
            for key in updates[0]
        }
        self.blocks += 1
        metrics["collected_rollouts"] = collected
        metrics["qsafe_optimizer_steps"] = nr_updates
        metrics["candidate_fallback_rate"] = fallback_count / action_count
        metrics["selected_safe_q"] = selected_q_sum / action_count
        metrics["epsilon_minus_selected_q"] = (
            self.epsilon - metrics["selected_safe_q"]
        )
        return metrics

    def _sample_actions(self, state):
        with torch.no_grad(), autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16
        ):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            actions, processed_actions, selected_q, fallback = sample_safe_actions(
                state,
                self.policy,
                self.critic.q,
                self.nr_action_samples,
                self.epsilon,
                selection="boundary",
            )
        return (
            actions.float().cpu().numpy(),
            processed_actions.float().cpu().numpy(),
            selected_q.float().cpu().numpy(),
            fallback.cpu().numpy(),
        )

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
            raise ValueError(
                "Safety environment info must provide a binary 'failure' label"
            )
        return actual_next_state, terminations, truncations, failures

    def _update_critic(self):
        states, next_states, actions, failures, terminations, truncations = (
            self.replay_buffer.sample(self.batch_size)
        )
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        failures = torch.as_tensor(
            failures, dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        terminations = torch.as_tensor(
            terminations, dtype=torch.float32, device=self.device
        )
        truncations = torch.as_tensor(
            truncations, dtype=torch.float32, device=self.device
        )

        with autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16
        ):
            with torch.no_grad():
                next_actions, _, _ = self.policy.get_action(next_states)
                next_q = self.critic.q_target(next_states, next_actions)
                target = qsafe_bellman_target(
                    failures,
                    torch.maximum(terminations, truncations),
                    next_q,
                    self.gamma,
                )
            safe_q = self.critic.q(states, actions)
            loss = F.mse_loss(safe_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1
        self._update_target()
        return {
            "qsafe_loss": loss.detach().item(),
            "safe_q": safe_q.detach().mean().item(),
            "safe_target": target.detach().mean().item(),
        }

    def _update_target(self):
        with torch.no_grad():
            for online, target in zip(
                self.critic.q.parameters(), self.critic.q_target.parameters()
            ):
                target.data.mul_(1.0 - self.tau).add_(online.data, alpha=self.tau)
