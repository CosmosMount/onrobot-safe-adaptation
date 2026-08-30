import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast

from algorithms.qsafe.pytorch.critic import get_critic as get_safe_critic
from algorithms.sac.pytorch.critic import get_critic as get_task_critic
from algorithms.sac.pytorch.entropy_coefficient import get_entropy_coefficient
from algorithms.sac.pytorch.policy import get_policy
from sqrl.sac.pytorch.replay_buffer import ReplayBuffer
from sqrl.sac.pytorch.rollout_buffer import RolloutBuffer


sqrl_pretrain_logger = logging.getLogger("sqrl_pretrain")


class SQRL_Pretrain_SAC:

    def __init__(self, config, env, device, safety_env=None):
        self.config = config
        self.env = env
        self.safety_env = safety_env if safety_env is not None else getattr(env, "safety_env", None)
        self.device = torch.device(device)

        self.seed = int(config.environment.seed)
        self.nr_envs = int(config.environment.nr_envs)
        self.compile_mode = config.algorithm.compile_mode
        self.bf16_mixed_precision_training = bool(
            config.algorithm.bf16_mixed_precision_training
        )
        self.gamma = float(config.algorithm.gamma)
        self.safe_gamma = float(config.algorithm.get("safe_gamma", 0.7))
        self.tau = float(config.algorithm.tau)
        self.safe_tau = float(config.algorithm.get("safe_tau", config.algorithm.tau))
        self.batch_size = int(config.algorithm.batch_size)
        self.safety_batch_size = int(config.algorithm.get("safety_batch_size", config.algorithm.batch_size))
        self.learning_starts = int(config.algorithm.get("learning_starts", 0))
        self.nr_pretrain_steps = int(
            config.algorithm.get("n_pre", config.algorithm.get("nr_epochs", 500_000))
        )
        self.nr_offline_steps = int(
            config.algorithm.get(
                "n_off", config.algorithm.get("nr_task_epochs", 1)
            )
        )
        self.nr_safety_rollouts = int(config.algorithm.get("k", config.algorithm.get("nr_safety_epochs", 1)))
        self.epsilon_safe = float(config.algorithm.get("epsilon_safe", 0.1))
        self.max_safe_action_samples = int(config.algorithm.get("max_safe_action_samples", 100))
        self.task_buffer_size = int(
            config.algorithm.get(
                "task_buffer_size", config.algorithm.get("buffer_size", 1_000_000)
            )
        )
        self.safety_buffer_size = int(config.algorithm.get("safety_buffer_size", 100_000))
        self.max_safety_trajectories = int(config.algorithm.get("max_safety_trajectories", self.nr_safety_rollouts))

        if self.nr_pretrain_steps < 1:
            raise ValueError("n_pre must be at least 1")
        if self.nr_offline_steps < 1:
            raise ValueError("n_off must be at least 1")
        if self.nr_safety_rollouts < 1:
            raise ValueError("k must be at least 1")
        if self.max_safe_action_samples < 1:
            raise ValueError("max_safe_action_samples must be at least 1")
        if self.bf16_mixed_precision_training and self.device.type != "cuda":
            raise ValueError("bfloat16 mixed precision training requires CUDA")

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.policy = get_policy(config, env, self.device)
        self.task_critic = get_task_critic(config, env, self.device)
        self.safe_critic = get_safe_critic(config, env, self.device)
        self.entropy_coefficient = get_entropy_coefficient(
            config, env, self.device
        )

        learning_rate = float(config.algorithm.get("learning_rate", 3e-4))
        policy_lr = float(config.algorithm.get("policy_lr", learning_rate))
        qtask_lr = float(config.algorithm.get("qtask_lr", learning_rate))
        qsafe_lr = float(config.algorithm.get("qsafe_lr", learning_rate))
        entropy_lr = float(config.algorithm.get("entropy_lr", learning_rate))
        optimizer_options = {"fused": True} if self.device.type == "cuda" else {}

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=policy_lr, **optimizer_options
        )
        self.qtask_optimizer = optim.Adam(
            list(self.task_critic.q1.parameters())
            + list(self.task_critic.q2.parameters()),
            lr=qtask_lr,
            **optimizer_options,
        )
        self.qsafe_optimizer = optim.Adam(
            self.safe_critic.q.parameters(), lr=qsafe_lr, **optimizer_options
        )
        self.entropy_optimizer = optim.Adam(
            [self.entropy_coefficient.log_alpha],
            lr=entropy_lr,
            **optimizer_options,
        )

        self.env_as_low = np.asarray(env.single_action_space.low, dtype=np.float32)
        self.env_as_high = np.asarray(env.single_action_space.high, dtype=np.float32)
        self.offline_replay_buffer = None
        self.safety_replay_buffer = None
        self.state = None
        self.task_steps = 0
        self.task_updates = 0
        self.safety_rollouts = 0
        self.safety_updates = 0

    def train(self):
        if self.safety_env is None or self.safety_env is self.env:
            raise ValueError("SQRL pre-training requires an independent safety_env")

        self.offline_replay_buffer = ReplayBuffer(
            capacity=self.task_buffer_size,
            nr_envs=self.nr_envs,
            os_shape=self.env.single_observation_space.shape,
            as_shape=self.env.single_action_space.shape,
            rng=self.rng,
            device=self.device,
        )

        self.safety_replay_buffer = RolloutBuffer(
            capacity=self.safety_buffer_size,
            nr_envs=self.nr_envs,
            os_shape=self.env.single_observation_space.shape,
            as_shape=self.env.single_action_space.shape,
            rng=self.rng,
            max_trajectories=self.max_safety_trajectories,
        )

        reset_result = self.env.reset()
        self.state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        pretrain_step = 0
        while self.task_steps < self.nr_pretrain_steps:
            for _ in range(self.nr_offline_steps):
                if self.task_steps >= self.nr_pretrain_steps:
                    break
                self.train_task()

            # Collect k complete on-policy rollouts first, then update QSafe
            # exactly once from D_safe.
            safety_metrics = self.train_safety()
            pretrain_step += 1
            sqrl_pretrain_logger.info(
                "pretrain_step=%d task_steps=%d task_updates=%d "
                "safety_rollouts=%d safety_updates=%d qsafe_loss=%.6f",
                pretrain_step,
                self.task_steps,
                self.task_updates,
                self.safety_rollouts,
                self.safety_updates,
                safety_metrics["qsafe_loss"],
            )

        return self.policy, self.safe_critic.q

    def train_task(self):
        def sample_actions():
            if self.task_steps < self.learning_starts:
                processed_actions = np.asarray(
                    [self.env.single_action_space.sample() for _ in range(self.nr_envs)], dtype=np.float32
                )
                actions = 2.0 * (processed_actions - self.env_as_low) / (self.env_as_high - self.env_as_low) - 1.0
                return actions, processed_actions

            with torch.no_grad(), autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training
            ):
                actions, processed_actions, _ = self.policy.get_action(
                    torch.as_tensor(self.state, dtype=torch.float32, device=self.device)
                )
            return actions.cpu().numpy(), processed_actions.cpu().numpy()

        def process_step(next_state, terminations, truncations, info):
            terminations = np.asarray(terminations, dtype=bool).reshape(self.nr_envs)
            truncations = np.asarray(truncations, dtype=bool).reshape(self.nr_envs)
            actual_next_state = np.asarray(next_state, dtype=np.float32).copy()
            for index, single_done in enumerate(terminations | truncations):
                if single_done and hasattr(self.env, "get_final_observation_at_index"):
                    actual_next_state[index] = np.asarray(
                        self.env.get_final_observation_at_index(info, index), dtype=np.float32
                    )
                elif single_done and isinstance(info, dict) and "final_observation" in info and info["final_observation"][index] is not None:
                    actual_next_state[index] = np.asarray(info["final_observation"][index], dtype=np.float32)

            if isinstance(info, dict) and "failure" in info:
                failures = np.asarray(info["failure"], dtype=np.float32)
            elif isinstance(info, dict) and "failures" in info:
                failures = np.asarray(info["failures"], dtype=np.float32)
            else:
                failures = np.zeros(self.nr_envs, dtype=np.float32)
            return actual_next_state, terminations, failures

        def update_networks():
            states, next_states, actions, rewards, terminations, _ = self.offline_replay_buffer.sample(self.batch_size)
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
                policy_loss = (self.entropy_coefficient().detach() * current_log_probs - current_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            entropy_loss = self.entropy_coefficient.loss(-current_log_probs.detach()).mean()
            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_optimizer.step()
            self.task_updates += 1
            return {
                "qtask_loss": qtask_loss.detach().item(),
                "policy_loss": policy_loss.detach().item(),
                "entropy_loss": entropy_loss.detach().item(),
            }

        self.policy.train()
        self.task_critic.q1.train()
        self.task_critic.q2.train()

        actions, processed_actions = sample_actions()
        next_state, rewards, terminations, truncations, info = self.env.step(processed_actions)
        actual_next_state, terminations, failures = process_step(next_state, terminations, truncations, info)
        self.offline_replay_buffer.add(
            np.asarray(self.state, dtype=np.float32), actual_next_state, actions, np.asarray(rewards, dtype=np.float32),
            terminations.astype(np.float32), failures
        )
        self.state = next_state
        self.task_steps += self.nr_envs

        if self.task_steps < self.learning_starts:
            return None
        return update_networks()

    def train_safety(self):
        def sample_actions(state):
            with torch.no_grad(), autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training
            ):
                state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                accepted = torch.zeros(self.nr_envs, dtype=torch.bool, device=self.device)
                best_safe_q = torch.full((self.nr_envs,), torch.inf, device=self.device)
                actions = torch.empty(
                    (self.nr_envs,) + self.safety_env.single_action_space.shape, dtype=torch.float32, device=self.device
                )
                processed_actions = torch.empty_like(actions)

                for _ in range(self.max_safe_action_samples):
                    candidate_actions, candidate_processed_actions, _ = self.policy.get_action(state)
                    candidate_safe_q = self.safe_critic.q(state, candidate_actions).reshape(self.nr_envs)
                    unresolved = ~accepted
                    better_candidate = unresolved & (candidate_safe_q < best_safe_q)
                    actions[better_candidate] = candidate_actions[better_candidate]
                    processed_actions[better_candidate] = candidate_processed_actions[better_candidate]
                    best_safe_q[better_candidate] = candidate_safe_q[better_candidate]
                    accepted |= unresolved & (candidate_safe_q < self.epsilon_safe)
                    if bool(accepted.all()):
                        break
            return actions.cpu().numpy(), processed_actions.cpu().numpy()

        def process_step(next_state, terminations, truncations, info):
            terminations = np.asarray(terminations, dtype=bool).reshape(self.nr_envs)
            truncations = np.asarray(truncations, dtype=bool).reshape(self.nr_envs)
            actual_next_state = np.asarray(next_state, dtype=np.float32).copy()
            for index, single_done in enumerate(terminations | truncations):
                if single_done and hasattr(self.safety_env, "get_final_observation_at_index"):
                    actual_next_state[index] = np.asarray(
                        self.safety_env.get_final_observation_at_index(info, index), dtype=np.float32
                    )
                elif single_done and isinstance(info, dict) and "final_observation" in info and info["final_observation"][index] is not None:
                    actual_next_state[index] = np.asarray(info["final_observation"][index], dtype=np.float32)

            if isinstance(info, dict) and "failure" in info:
                failures = np.asarray(info["failure"], dtype=np.float32)
            elif isinstance(info, dict) and "failures" in info:
                failures = np.asarray(info["failures"], dtype=np.float32)
            else:
                raise ValueError("safety_env info must provide a binary 'failure' label")
            return actual_next_state, terminations, truncations, failures

        def update_qsafe():
            states, next_states, actions, failures, terminations, _ = self.safety_replay_buffer.sample(
                self.safety_batch_size
            )
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            next_failures = torch.as_tensor(failures, dtype=torch.float32, device=self.device).reshape(-1, 1)
            terminations = torch.as_tensor(terminations, dtype=torch.float32, device=self.device)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    next_actions, _, _ = self.policy.get_action(next_states)
                    next_safe_q = self.safe_critic.q_target(next_states, next_actions)
                    safe_target = self.safe_gamma * (
                        next_failures + (1.0 - next_failures) * (1.0 - terminations.reshape(-1, 1)) * next_safe_q
                    )
                safe_q = self.safe_critic.q(states, actions)
                qsafe_loss = F.mse_loss(safe_q, safe_target)

            self.qsafe_optimizer.zero_grad()
            qsafe_loss.backward()
            self.qsafe_optimizer.step()
            self.safety_updates += 1

            with torch.no_grad():
                for online_parameter, target_parameter in zip(
                    self.safe_critic.q.parameters(), self.safe_critic.q_target.parameters()
                ):
                    target_parameter.data.mul_(1.0 - self.safe_tau).add_(online_parameter.data, alpha=self.safe_tau)
            return {
                "qsafe_loss": qsafe_loss.detach().item(),
                "safe_q": safe_q.detach().mean().item(),
                "safe_target": safe_target.detach().mean().item(),
            }

        self.policy.eval()
        self.safe_critic.q.train()

        reset_result = self.safety_env.reset()
        safety_state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        collected_rollouts = 0
        while collected_rollouts < self.nr_safety_rollouts:
            actions, processed_actions = sample_actions(safety_state)
            next_state, _, terminations, truncations, info = self.safety_env.step(processed_actions)
            actual_next_state, terminations, truncations, failures = process_step(
                next_state, terminations, truncations, info
            )
            completed_now = self.safety_replay_buffer.add(
                np.asarray(safety_state, dtype=np.float32), actual_next_state, actions, failures, terminations,
                truncations, self.nr_safety_rollouts - collected_rollouts
            )
            collected_rollouts += completed_now
            self.safety_rollouts += completed_now
            safety_state = next_state

        metrics = update_qsafe()
        metrics["collected_rollouts"] = collected_rollouts
        return metrics
