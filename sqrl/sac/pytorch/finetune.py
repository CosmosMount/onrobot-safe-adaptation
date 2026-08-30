import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast

from algorithms.sac.pytorch.critic import get_critic as get_task_critic
from algorithms.sac.pytorch.entropy_coefficient import get_entropy_coefficient
from sqrl.sac.pytorch.replay_buffer import ReplayBuffer


sqrl_finetune_logger = logging.getLogger("sqrl_finetune")


class SQRLFinetuner:

    def __init__(self, config, target_env, policy, qsafe, device):
        self.config = config
        self.target_env = target_env
        self.policy = policy
        self.qsafe = qsafe
        self.device = torch.device(device)

        self.seed = int(config.environment.seed)
        self.nr_envs = int(config.environment.nr_envs)
        self.bf16_mixed_precision_training = bool(config.algorithm.bf16_mixed_precision_training)
        self.gamma = float(config.algorithm.gamma)
        self.tau = float(config.algorithm.tau)
        self.batch_size = int(config.algorithm.batch_size)
        self.nr_target_steps = int(config.algorithm.get("n_target", config.algorithm.get("total_timesteps", 500_000)))
        self.task_utd_ratio = float(config.algorithm.get("task_utd_ratio", 1.0))
        self.epsilon_safe = float(config.algorithm.get("epsilon_safe", 0.1))
        self.nr_action_candidates = int(config.algorithm.get("max_safe_action_samples", 100))
        self.buffer_size = int(config.algorithm.get("task_buffer_size", config.algorithm.get("buffer_size", 1_000_000)))
        self.logging_frequency = max(1, int(config.algorithm.get("logging_frequency", 1)))

        if self.nr_target_steps < 1:
            raise ValueError("n_target must be at least 1")
        if self.nr_action_candidates < 1:
            raise ValueError("max_safe_action_samples must be at least 1")
        if not np.isfinite(self.task_utd_ratio) or self.task_utd_ratio < 0.0:
            raise ValueError("task_utd_ratio must be a finite non-negative value")
        if self.bf16_mixed_precision_training and self.device.type != "cuda":
            raise ValueError("bfloat16 mixed precision training requires CUDA")

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.task_critic = get_task_critic(config, target_env, self.device)
        self.entropy_coefficient = get_entropy_coefficient(config, target_env, self.device)
        self.qsafe.eval()
        for parameter in self.qsafe.parameters():
            parameter.requires_grad_(False)

        learning_rate = float(config.algorithm.get("learning_rate", 3e-4))
        policy_lr = float(config.algorithm.get("policy_lr", learning_rate))
        qtask_lr = float(config.algorithm.get("qtask_lr", learning_rate))
        entropy_lr = float(config.algorithm.get("entropy_lr", learning_rate))
        dual_lr = float(config.algorithm.get("dual_learning_rate", learning_rate))
        optimizer_options = {"fused": True} if self.device.type == "cuda" else {}

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr, **optimizer_options)
        self.qtask_optimizer = optim.Adam(
            list(self.task_critic.q1.parameters()) + list(self.task_critic.q2.parameters()),
            lr=qtask_lr, **optimizer_options
        )
        self.entropy_optimizer = optim.Adam(
            [self.entropy_coefficient.log_alpha], lr=entropy_lr, **optimizer_options
        )
        self.nu = torch.tensor(
            float(config.algorithm.get("initial_nu", 0.0)), dtype=torch.float32,
            device=self.device, requires_grad=True
        )
        self.dual_optimizer = optim.Adam([self.nu], lr=dual_lr, **optimizer_options)

        self.replay_buffer = None
        self.state = None
        self.target_steps = 0
        self.updates = 0
        self.task_update_credit = 0.0

    def train(self):
        def sample_actions(state):
            with torch.no_grad(), autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training
            ):
                state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                candidate_states = state[:, None, :].expand(-1, self.nr_action_candidates, -1)
                flat_states = candidate_states.reshape(self.nr_envs * self.nr_action_candidates, -1)
                actions, processed_actions, log_probs = self.policy.get_action(flat_states)
                actions = actions.reshape((self.nr_envs, self.nr_action_candidates) + self.target_env.single_action_space.shape)
                processed_actions = processed_actions.reshape_as(actions)
                log_probs = log_probs.reshape(self.nr_envs, self.nr_action_candidates)
                safe_q = self.qsafe(flat_states, actions.reshape(self.nr_envs * self.nr_action_candidates, -1))
                safe_q = safe_q.reshape(self.nr_envs, self.nr_action_candidates)
                # Eq. 3: sample from the task policy restricted to QSafe < epsilon.
                safe_mask = safe_q < self.epsilon_safe
                fallback = ~safe_mask.any(dim=1)
                masked_logits = torch.where(safe_mask, log_probs, torch.full_like(log_probs, -torch.inf))
                masked_logits = torch.where(fallback[:, None], torch.zeros_like(masked_logits), masked_logits)
                selected = torch.distributions.Categorical(logits=masked_logits).sample()
                selected = torch.where(fallback, safe_q.argmin(dim=1), selected)
                indices = torch.arange(self.nr_envs, device=self.device)
            return (
                actions[indices, selected].float().cpu().numpy(),
                processed_actions[indices, selected].float().cpu().numpy(),
                safe_q[indices, selected].float().cpu().numpy(),
                fallback.cpu().numpy(),
            )

        def process_step(next_state, terminations, truncations, info):
            terminations = np.asarray(terminations, dtype=bool).reshape(self.nr_envs)
            truncations = np.asarray(truncations, dtype=bool).reshape(self.nr_envs)
            actual_next_state = np.asarray(next_state, dtype=np.float32).copy()
            for index, single_done in enumerate(terminations | truncations):
                if single_done and hasattr(self.target_env, "get_final_observation_at_index"):
                    actual_next_state[index] = np.asarray(
                        self.target_env.get_final_observation_at_index(info, index), dtype=np.float32
                    )
                elif single_done and isinstance(info, dict) and "final_observation" in info and info["final_observation"][index] is not None:
                    actual_next_state[index] = np.asarray(info["final_observation"][index], dtype=np.float32)

            if isinstance(info, dict) and "failure" in info:
                failures = np.asarray(info["failure"], dtype=np.float32).reshape(self.nr_envs)
            elif isinstance(info, dict) and "failures" in info:
                failures = np.asarray(info["failures"], dtype=np.float32).reshape(self.nr_envs)
            else:
                raise ValueError("target_env info must provide a binary 'failure' label")
            if not np.all((failures == 0.0) | (failures == 1.0)):
                raise ValueError("target_env failure labels must be binary")
            if np.any(failures.astype(bool) & ~terminations):
                raise ValueError("Every target-task failure must set terminated=True")
            return actual_next_state, terminations, failures

        def update_networks():
            states, next_states, actions, rewards, terminations, _ = self.replay_buffer.sample(self.batch_size)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    next_actions, _, next_log_probs = self.policy.get_action(next_states)
                    next_q = torch.minimum(
                        self.task_critic.q1_target(next_states, next_actions),
                        self.task_critic.q2_target(next_states, next_actions),
                    )
                    target_q = rewards.reshape(-1, 1) + self.gamma * (1.0 - terminations.reshape(-1, 1)) * (
                        next_q - self.entropy_coefficient().detach() * next_log_probs
                    )
                q1 = self.task_critic.q1(states, actions)
                q2 = self.task_critic.q2(states, actions)
                qtask_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))

            self.qtask_optimizer.zero_grad()
            qtask_loss.backward()
            self.qtask_optimizer.step()

            with torch.no_grad():
                for online_parameter, target_parameter in zip(
                    self.task_critic.q1.parameters(), self.task_critic.q1_target.parameters()
                ):
                    target_parameter.data.mul_(1.0 - self.tau).add_(online_parameter.data, alpha=self.tau)
                for online_parameter, target_parameter in zip(
                    self.task_critic.q2.parameters(), self.task_critic.q2_target.parameters()
                ):
                    target_parameter.data.mul_(1.0 - self.tau).add_(online_parameter.data, alpha=self.tau)

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                current_actions, _, current_log_probs = self.policy.get_action(states)
                current_q = torch.minimum(
                    self.task_critic.q1(states, current_actions), self.task_critic.q2(states, current_actions)
                )
                safety_q = self.qsafe(states, current_actions)
                # Minimize the negative of Eq. 4; QSafe stays differentiable only through the action.
                policy_loss = (
                    self.entropy_coefficient().detach() * current_log_probs - current_q
                    + self.nu.detach() * (safety_q - self.epsilon_safe)
                ).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            entropy_loss = self.entropy_coefficient.loss(-current_log_probs.detach()).mean()
            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_optimizer.step()

            # Algorithm 2 minimizes Eq. 4 over the non-negative safety multiplier nu.
            dual_loss = (self.nu * (self.epsilon_safe - safety_q.detach())).mean()
            self.dual_optimizer.zero_grad()
            dual_loss.backward()
            self.dual_optimizer.step()
            with torch.no_grad():
                self.nu.clamp_(min=0.0)

            self.updates += 1
            return {
                "qtask_loss": qtask_loss.detach().item(),
                "policy_loss": policy_loss.detach().item(),
                "entropy_loss": entropy_loss.detach().item(),
                "dual_loss": dual_loss.detach().item(),
                "safe_q": safety_q.detach().mean().item(),
                "nu": self.nu.detach().item(),
            }

        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            nr_envs=self.nr_envs,
            os_shape=self.target_env.single_observation_space.shape,
            as_shape=self.target_env.single_action_space.shape,
            rng=self.rng,
            device=self.device,
        )
        self.policy.train()
        self.task_critic.q1.train()
        self.task_critic.q2.train()
        self.qsafe.eval()

        reset_result = self.target_env.reset()
        self.state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        while self.target_steps < self.nr_target_steps:
            actions, processed_actions, selected_safe_q, fallback = sample_actions(self.state)
            next_state, rewards, terminations, truncations, info = self.target_env.step(processed_actions)
            actual_next_state, terminations, failures = process_step(next_state, terminations, truncations, info)
            self.replay_buffer.add(
                np.asarray(self.state, dtype=np.float32), actual_next_state, actions,
                np.asarray(rewards, dtype=np.float32), terminations.astype(np.float32), failures
            )
            self.state = next_state
            self.target_steps += self.nr_envs
            self.task_update_credit += self.nr_envs * self.task_utd_ratio
            nr_updates = int(np.floor(self.task_update_credit + 1e-12))
            self.task_update_credit -= nr_updates
            metrics = None
            for _ in range(nr_updates):
                metrics = update_networks()

            if metrics is not None and self.target_steps % self.logging_frequency < self.nr_envs:
                sqrl_finetune_logger.info(
                    "target_steps=%d updates=%d failure_rate=%.6f fallback_rate=%.6f "
                    "selected_safe_q=%.6f policy_loss=%.6f nu=%.6f",
                    self.target_steps, self.updates, failures.mean(), fallback.mean(), selected_safe_q.mean(),
                    metrics["policy_loss"], metrics["nu"]
                )

        return self.policy
