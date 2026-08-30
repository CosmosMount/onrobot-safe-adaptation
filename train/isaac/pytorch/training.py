"""PyTorch SQRL source pre-training on Isaac Lab."""
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast

from algorithms.sac.pytorch.critic import get_critic as get_task_critic
from algorithms.sac.pytorch.entropy_coefficient import get_entropy_coefficient
from sqrl.sac.pytorch.replay_buffer import ReplayBuffer
from sqrl.sac.pytorch.rollout_buffer import RolloutBuffer
from train.common.pytorch import SQRLTrainingBase


sqrl_training_logger = logging.getLogger("sqrl_training")


class IsaacTraining(SQRLTrainingBase):
    def pretrain(self):
        algorithm = self.config.algorithm
        nr_task_envs = int(self.env.nr_task_envs)
        nr_safety_envs = int(self.env.nr_safety_envs)
        if not hasattr(self.env, "step_partitions") or nr_task_envs < 1 or nr_safety_envs < 1:
            raise ValueError("source pretraining requires non-empty task/safety partitions")
        if int(self.config.environment.nr_envs) != nr_task_envs + nr_safety_envs:
            raise ValueError("nr_task_envs + nr_safety_envs must equal nr_envs")

        gamma = float(algorithm.gamma)
        safe_gamma = float(algorithm.get("safe_gamma", 0.7))
        tau = float(algorithm.tau)
        safe_tau = float(algorithm.get("safe_tau", algorithm.tau))
        batch_size = int(algorithm.batch_size)
        safety_batch_size = int(algorithm.get("safety_batch_size", algorithm.batch_size))
        learning_starts = int(algorithm.get("learning_starts", 0))
        total_timesteps = int(algorithm.get("n_pre", algorithm.get("total_timesteps", 500_000)))
        task_utd_ratio = float(algorithm.get("task_utd_ratio", 1.0))
        k = int(algorithm.get("k", nr_safety_envs))
        epsilon_safe = float(algorithm.get("epsilon_safe", 0.1))
        nr_candidates = int(algorithm.get("max_safe_action_samples", 100))
        bf16 = bool(algorithm.bf16_mixed_precision_training)
        if total_timesteps < 1 or k < 1 or nr_candidates < 1:
            raise ValueError("n_pre, k and max_safe_action_samples must be positive")
        if not np.isfinite(task_utd_ratio) or task_utd_ratio < 0.0:
            raise ValueError("task_utd_ratio must be a finite non-negative value")
        if bf16 and self.device.type != "cuda":
            raise ValueError("bfloat16 mixed precision training requires CUDA")

        task_critic = get_task_critic(self.config, self.env, self.device)
        entropy_coefficient = get_entropy_coefficient(self.config, self.env, self.device)
        learning_rate = float(algorithm.get("learning_rate", 3e-4))
        optimizer_options = {"fused": True} if self.device.type == "cuda" else {}
        policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=float(algorithm.get("policy_lr", learning_rate)), **optimizer_options
        )
        qtask_optimizer = optim.Adam(
            list(task_critic.q1.parameters()) + list(task_critic.q2.parameters()),
            lr=float(algorithm.get("qtask_lr", learning_rate)), **optimizer_options
        )
        qsafe_optimizer = optim.Adam(
            self.safe_critic.q.parameters(), lr=float(algorithm.get("qsafe_lr", learning_rate)), **optimizer_options
        )
        entropy_optimizer = optim.Adam(
            [entropy_coefficient.log_alpha], lr=float(algorithm.get("entropy_lr", learning_rate)), **optimizer_options
        )
        task_replay = ReplayBuffer(
            int(algorithm.get("task_buffer_size", algorithm.get("buffer_size", 1_000_000))), nr_task_envs,
            self.env.single_observation_space.shape, self.env.single_action_space.shape, self.rng, self.device
        )
        safety_replay = RolloutBuffer(
            int(algorithm.get("safety_buffer_size", 100_000)), nr_safety_envs,
            self.env.single_observation_space.shape, self.env.single_action_space.shape, self.rng,
            max_trajectories=max(k, int(algorithm.get("max_safety_trajectories", k)))
        )

        def final_observations(step, nr_envs):
            terminations = np.asarray(step.terminated, dtype=bool).reshape(nr_envs)
            truncations = np.asarray(step.truncated, dtype=bool).reshape(nr_envs)
            next_states = np.asarray(step.observation, dtype=np.float32).copy()
            for index in np.flatnonzero(terminations | truncations):
                final_state = step.info.get("final_observation", [None] * nr_envs)[index]
                if final_state is not None:
                    next_states[index] = np.asarray(final_state, dtype=np.float32)
            failures = np.asarray(step.info.get("failure"), dtype=np.float32).reshape(nr_envs)
            if not np.all((failures == 0.0) | (failures == 1.0)):
                raise ValueError("source environment failure labels must be binary")
            if np.any(failures.astype(bool) & ~terminations):
                raise ValueError("Every failure must set terminated=True")
            return next_states, terminations, truncations, failures

        def sample_safe_actions(states):
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                repeated_states = states[:, None, :].expand(-1, nr_candidates, -1)
                flat_states = repeated_states.reshape(nr_safety_envs * nr_candidates, -1)
                actions, processed_actions, _ = self.policy.get_action(flat_states)
                safe_q = self.safe_critic.q(flat_states, actions).reshape(nr_safety_envs, nr_candidates)
                actions = actions.reshape((nr_safety_envs, nr_candidates) + self.env.single_action_space.shape)
                processed_actions = processed_actions.reshape_as(actions)
                safe_mask = safe_q < epsilon_safe
                fallback = ~safe_mask.any(dim=1)
                selected = torch.where(
                    safe_mask, safe_q, torch.full_like(safe_q, -torch.inf)
                ).argmax(dim=1)
                selected = torch.where(fallback, safe_q.argmin(dim=1), selected)
                indices = torch.arange(nr_safety_envs, device=self.device)
            return actions[indices, selected].float().cpu().numpy(), processed_actions[indices, selected].float().cpu().numpy()

        def update_task():
            states, next_states, actions, rewards, terminations, _ = task_replay.sample(batch_size)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                with torch.no_grad():
                    next_actions, _, next_log_probs = self.policy.get_action(next_states)
                    next_q = torch.minimum(
                        task_critic.q1_target(next_states, next_actions), task_critic.q2_target(next_states, next_actions)
                    )
                    target_q = rewards.reshape(-1, 1) + gamma * (1.0 - terminations.reshape(-1, 1)) * (
                        next_q - entropy_coefficient().detach() * next_log_probs
                    )
                q1, q2 = task_critic.q1(states, actions), task_critic.q2(states, actions)
                qtask_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))
            qtask_optimizer.zero_grad()
            qtask_loss.backward()
            qtask_optimizer.step()
            with torch.no_grad():
                for online, target in zip(task_critic.q1.parameters(), task_critic.q1_target.parameters()):
                    target.data.mul_(1.0 - tau).add_(online.data, alpha=tau)
                for online, target in zip(task_critic.q2.parameters(), task_critic.q2_target.parameters()):
                    target.data.mul_(1.0 - tau).add_(online.data, alpha=tau)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                current_actions, _, current_log_probs = self.policy.get_action(states)
                current_q = torch.minimum(task_critic.q1(states, current_actions), task_critic.q2(states, current_actions))
                policy_loss = (entropy_coefficient().detach() * current_log_probs - current_q).mean()
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            entropy_loss = entropy_coefficient.loss(-current_log_probs.detach()).mean()
            entropy_optimizer.zero_grad()
            entropy_loss.backward()
            entropy_optimizer.step()
            return qtask_loss.detach().item(), policy_loss.detach().item()

        def update_safe():
            states, next_states, actions, failures, terminations, _ = safety_replay.sample(safety_batch_size)
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            failures = torch.as_tensor(failures, dtype=torch.float32, device=self.device).reshape(-1, 1)
            terminations = torch.as_tensor(terminations, dtype=torch.float32, device=self.device).reshape(-1, 1)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                with torch.no_grad():
                    next_actions, _, _ = self.policy.get_action(next_states)
                    next_safe_q = self.safe_critic.q_target(next_states, next_actions)
                    safe_target = safe_gamma * (
                        failures + (1.0 - failures) * (1.0 - terminations) * next_safe_q
                    )
                safe_q = self.safe_critic.q(states, actions)
                qsafe_loss = F.mse_loss(safe_q, safe_target)
            qsafe_optimizer.zero_grad()
            qsafe_loss.backward()
            qsafe_optimizer.step()
            with torch.no_grad():
                for online, target in zip(self.safe_critic.q.parameters(), self.safe_critic.q_target.parameters()):
                    target.data.mul_(1.0 - safe_tau).add_(online.data, alpha=safe_tau)
            return qsafe_loss.detach().item()

        self.policy.train()
        self.safe_critic.q.train()
        task_critic.q1.train()
        task_critic.q2.train()
        task_states, safety_states = self.env.reset_partitions()
        task_steps = task_updates = safety_rollouts = safety_updates = 0
        safety_in_block = 0
        update_credit = 0.0
        last_task_loss = last_policy_loss = last_safe_loss = float("nan")
        logging_frequency = max(1, int(algorithm.get("logging_frequency", 50_000)))
        next_log = logging_frequency
        sqrl_training_logger.info(
            "source rollout=partitioned: task and safety pools advance together; QSafe updates once per exact k=%d complete trajectories",
            k,
        )
        while task_steps < total_timesteps:
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                task_tensor = torch.as_tensor(task_states, dtype=torch.float32, device=self.device)
                if task_steps < learning_starts:
                    raw_actions = torch.as_tensor(
                        np.asarray([self.env.single_action_space.sample() for _ in range(nr_task_envs)]),
                        dtype=torch.float32, device=self.device
                    )
                    task_actions = self.policy.project(task_tensor, raw_actions)
                    task_processed = self.policy.env_low + 0.5 * (task_actions + 1.0) * (
                        self.policy.env_high - self.policy.env_low
                    )
                else:
                    task_actions, task_processed, _ = self.policy.get_action(task_tensor)
            safety_actions, safety_processed = sample_safe_actions(safety_states)
            task_step, safety_step = self.env.step_partitions(
                task_processed.float().cpu().numpy(), safety_processed
            )
            task_next, task_terminations, _, task_failures = final_observations(task_step, nr_task_envs)
            safety_next, safety_terminations, safety_truncations, safety_failures = final_observations(
                safety_step, nr_safety_envs
            )
            task_applied = np.asarray(task_step.info.get("applied_action", task_actions.float().cpu().numpy()), dtype=np.float32)
            safety_applied = np.asarray(safety_step.info.get("applied_action", safety_actions), dtype=np.float32)
            task_replay.add(
                np.asarray(task_states, dtype=np.float32), task_next, task_applied,
                np.asarray(task_step.reward, dtype=np.float32), task_terminations.astype(np.float32), task_failures
            )
            completed = safety_replay.add(
                np.asarray(safety_states, dtype=np.float32), safety_next, safety_applied, safety_failures,
                safety_terminations, safety_truncations, k - safety_in_block
            )
            safety_in_block += completed
            safety_rollouts += completed
            if safety_in_block == k:
                last_safe_loss = update_safe()
                safety_updates += 1
                safety_in_block = 0
            previous_steps = task_steps
            task_steps += nr_task_envs
            eligible_before = max(0, previous_steps - learning_starts)
            eligible_after = max(0, task_steps - learning_starts)
            update_credit += (eligible_after - eligible_before) * task_utd_ratio
            nr_updates = int(np.floor(update_credit + 1e-12))
            update_credit -= nr_updates
            for _ in range(nr_updates):
                last_task_loss, last_policy_loss = update_task()
            task_updates += nr_updates
            task_states, safety_states = task_step.observation, safety_step.observation
            if task_steps >= next_log or task_steps >= total_timesteps:
                sqrl_training_logger.info(
                    "task_steps=%d task_updates=%d safety_rollouts=%d safety_updates=%d qtask_loss=%.6f policy_loss=%.6f qsafe_loss=%.6f",
                    task_steps, task_updates, safety_rollouts, safety_updates,
                    last_task_loss, last_policy_loss, last_safe_loss
                )
                while next_log <= task_steps:
                    next_log += logging_frequency
        return self.policy, self.safe_critic.q
