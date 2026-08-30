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
from sqrl.pytorch.replay_buffer import ReplayBuffer


sqrl_pretrain_logger = logging.getLogger("sqrl_pretrain")


class SQRL_Pretrain_SAC:

    def __init__(self, config, env, device):
        self.config = config
        self.env = env
        self.device = torch.device(device)

        self.seed = int(config.environment.seed)
        self.nr_envs = int(config.environment.nr_envs)
        self.compile_mode = config.algorithm.compile_mode
        self.bf16_mixed_precision_training = bool(
            config.algorithm.bf16_mixed_precision_training
        )
        self.gamma = float(config.algorithm.gamma)
        self.safe_gamma = float(
            config.algorithm.get("safe_gamma", config.algorithm.gamma)
        )
        self.tau = float(config.algorithm.tau)
        self.safe_tau = float(config.algorithm.get("safe_tau", config.algorithm.tau))
        self.batch_size = int(config.algorithm.batch_size)
        self.safety_batch_size = int(
            config.algorithm.get("safety_batch_size", config.algorithm.batch_size)
        )
        self.learning_starts = int(config.algorithm.get("learning_starts", 0))
        self.nr_pretrain_steps = int(
            config.algorithm.get("n_pre", config.algorithm.get("nr_epochs", 1))
        )
        self.nr_offline_steps = int(
            config.algorithm.get(
                "n_off", config.algorithm.get("nr_task_epochs", 1)
            )
        )
        self.nr_safety_rollouts = int(
            config.algorithm.get("k", config.algorithm.get("nr_safety_epochs", 1))
        )
        self.task_buffer_size = int(
            config.algorithm.get(
                "task_buffer_size", config.algorithm.get("buffer_size", 1_000_000)
            )
        )
        self.safety_buffer_size = int(
            config.algorithm.get("safety_buffer_size", 100_000)
        )

        if self.nr_pretrain_steps < 1:
            raise ValueError("n_pre must be at least 1")
        if self.nr_offline_steps < 1:
            raise ValueError("n_off must be at least 1")
        if self.nr_safety_rollouts < 1:
            raise ValueError("k must be at least 1")
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
        self.offline_replay_buffer = ReplayBuffer(
            capacity=self.task_buffer_size,
            nr_envs=self.nr_envs,
            os_shape=self.env.single_observation_space.shape,
            as_shape=self.env.single_action_space.shape,
            rng=self.rng,
            device=self.device,
        )
        self.safety_replay_buffer = ReplayBuffer(
            capacity=self.safety_buffer_size,
            nr_envs=self.nr_envs,
            os_shape=self.env.single_observation_space.shape,
            as_shape=self.env.single_action_space.shape,
            rng=self.rng,
            device=self.device,
        )

        reset_result = self.env.reset()
        self.state = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        for pretrain_step in range(self.nr_pretrain_steps):
            for _ in range(self.nr_offline_steps):
                self.train_task()

            # Collect k complete on-policy rollouts first, then update QSafe
            # exactly once from D_safe.
            safety_metrics = self.train_safety()
            sqrl_pretrain_logger.info(
                "pretrain_step=%d task_steps=%d task_updates=%d "
                "safety_rollouts=%d safety_updates=%d qsafe_loss=%.6f",
                pretrain_step + 1,
                self.task_steps,
                self.task_updates,
                self.safety_rollouts,
                self.safety_updates,
                safety_metrics["qsafe_loss"],
            )

        return self.policy, self.safe_critic.q

    def train_task(self):
        self.policy.train()
        self.task_critic.q1.train()
        self.task_critic.q2.train()

        if self.task_steps < self.learning_starts:
            processed_actions = np.asarray(
                [self.env.single_action_space.sample() for _ in range(self.nr_envs)],
                dtype=np.float32,
            )
            actions = ( 2.0 * (processed_actions - self.env_as_low) / (self.env_as_high - self.env_as_low) - 1.0)
        else:
            with torch.no_grad(), autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.bf16_mixed_precision_training,
            ):
                actions, processed_actions, _ = self.policy.get_action(
                    torch.as_tensor(
                        self.state, dtype=torch.float32, device=self.device
                    )
                )
            actions = actions.cpu().numpy()
            processed_actions = processed_actions.cpu().numpy()

        next_state, rewards, terminations, truncations, info = self.env.step(
            processed_actions
        )
        terminations = np.asarray(terminations, dtype=bool).reshape(self.nr_envs)
        truncations = np.asarray(truncations, dtype=bool).reshape(self.nr_envs)
        done = terminations | truncations
        actual_next_state = np.asarray(next_state, dtype=np.float32).copy()
        for index, single_done in enumerate(done):
            if single_done and hasattr(self.env, "get_final_observation_at_index"):
                actual_next_state[index] = np.asarray(
                    self.env.get_final_observation_at_index(info, index),
                    dtype=np.float32,
                )
            elif (
                single_done
                and isinstance(info, dict)
                and "final_observation" in info
                and info["final_observation"][index] is not None
            ):
                actual_next_state[index] = np.asarray(
                    info["final_observation"][index], dtype=np.float32
                )

        if isinstance(info, dict) and "failure" in info:
            failures = np.asarray(info["failure"], dtype=np.float32)
        elif isinstance(info, dict) and "failures" in info:
            failures = np.asarray(info["failures"], dtype=np.float32)
        else:
            failures = terminations.astype(np.float32)

        self.offline_replay_buffer.add(
            np.asarray(self.state, dtype=np.float32),
            actual_next_state,
            actions,
            np.asarray(rewards, dtype=np.float32),
            terminations.astype(np.float32),
            failures,
        )
        self.state = next_state
        self.task_steps += self.nr_envs

        if self.task_steps < self.learning_starts:
            return None

        states, next_states, actions, rewards, terminations, _ = (
            self.offline_replay_buffer.sample(self.batch_size)
        )
        with autocast( device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
            with torch.no_grad():
                next_actions, _, next_log_probs = self.policy.get_action(next_states)
                next_q = torch.minimum(
                    self.task_critic.q1_target(next_states, next_actions),
                    self.task_critic.q2_target(next_states, next_actions),
                )
                target_q = rewards.reshape(-1, 1) + self.gamma * (
                    1.0 - terminations.reshape(-1, 1)
                ) * (
                    next_q
                    - self.entropy_coefficient().detach() * next_log_probs
                )

            q1 = self.task_critic.q1(states, actions)
            q2 = self.task_critic.q2(states, actions)
            qtask_loss = 0.5 * (
                F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            )

        self.qtask_optimizer.zero_grad()
        qtask_loss.backward()
        self.qtask_optimizer.step()

        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                self.task_critic.q1.parameters(),
                self.task_critic.q1_target.parameters(),
            ):
                target_parameter.data.mul_(1.0 - self.tau).add_(
                    online_parameter.data, alpha=self.tau
                )
            for online_parameter, target_parameter in zip(
                self.task_critic.q2.parameters(),
                self.task_critic.q2_target.parameters(),
            ):
                target_parameter.data.mul_(1.0 - self.tau).add_(
                    online_parameter.data, alpha=self.tau
                )

        with autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.bf16_mixed_precision_training,
        ):
            current_actions, _, current_log_probs = self.policy.get_action(states)
            current_q = torch.minimum(
                self.task_critic.q1(states, current_actions),
                self.task_critic.q2(states, current_actions),
            )
            policy_loss = (
                self.entropy_coefficient().detach() * current_log_probs - current_q
            ).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        entropy = -current_log_probs.detach()
        entropy_loss = self.entropy_coefficient.loss(entropy).mean()
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()
        self.task_updates += 1

        return {
            "qtask_loss": qtask_loss.detach().item(),
            "policy_loss": policy_loss.detach().item(),
            "entropy_loss": entropy_loss.detach().item(),
        }

    def train_safety(self):
        self.policy.eval()
        self.safe_critic.q.train()

        reset_result = self.env.reset()
        safety_state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        collected_rollouts = 0

        while collected_rollouts < self.nr_safety_rollouts:
            with torch.no_grad(), autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.bf16_mixed_precision_training,
            ):
                actions, processed_actions, _ = self.policy.get_action(
                    torch.as_tensor(
                        safety_state, dtype=torch.float32, device=self.device
                    )
                )
            actions = actions.cpu().numpy()
            processed_actions = processed_actions.cpu().numpy()

            next_state, rewards, terminations, truncations, info = self.env.step(
                processed_actions
            )
            terminations = np.asarray(terminations, dtype=bool).reshape(self.nr_envs)
            truncations = np.asarray(truncations, dtype=bool).reshape(self.nr_envs)
            done = terminations | truncations
            actual_next_state = np.asarray(next_state, dtype=np.float32).copy()
            for index, single_done in enumerate(done):
                if single_done and hasattr(
                    self.env, "get_final_observation_at_index"
                ):
                    actual_next_state[index] = np.asarray(
                        self.env.get_final_observation_at_index(info, index),
                        dtype=np.float32,
                    )
                elif (
                    single_done
                    and isinstance(info, dict)
                    and "final_observation" in info
                    and info["final_observation"][index] is not None
                ):
                    actual_next_state[index] = np.asarray(
                        info["final_observation"][index], dtype=np.float32
                    )

            if isinstance(info, dict) and "failure" in info:
                failures = np.asarray(info["failure"], dtype=np.float32)
            elif isinstance(info, dict) and "failures" in info:
                failures = np.asarray(info["failures"], dtype=np.float32)
            else:
                failures = terminations.astype(np.float32)

            self.safety_replay_buffer.add(
                np.asarray(safety_state, dtype=np.float32),
                actual_next_state,
                actions,
                np.asarray(rewards, dtype=np.float32),
                terminations.astype(np.float32),
                failures,
            )
            safety_state = next_state
            completed_now = int(done.sum())
            collected_rollouts += completed_now
            self.safety_rollouts += completed_now

        states, next_states, actions, _, terminations, failures = (
            self.safety_replay_buffer.sample(self.safety_batch_size)
        )
        with autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.bf16_mixed_precision_training,
        ):
            with torch.no_grad():
                next_actions, _, _ = self.policy.get_action(next_states)
                next_safe_q = self.safe_critic.q_target(next_states, next_actions)
                failures = failures.reshape(-1, 1)
                safe_target = failures + self.safe_gamma * (1.0 - failures) * (
                    1.0 - terminations.reshape(-1, 1)
                ) * next_safe_q

            safe_q = self.safe_critic.q(states, actions)
            qsafe_loss = F.mse_loss(safe_q, safe_target)

        # One and only one QSafe update after k on-policy rollouts.
        self.qsafe_optimizer.zero_grad()
        qsafe_loss.backward()
        self.qsafe_optimizer.step()
        self.safety_updates += 1

        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                self.safe_critic.q.parameters(),
                self.safe_critic.q_target.parameters(),
            ):
                target_parameter.data.mul_(1.0 - self.safe_tau).add_(
                    online_parameter.data, alpha=self.safe_tau
                )

        reset_result = self.env.reset()
        self.state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        return {
            "qsafe_loss": qsafe_loss.detach().item(),
            "safe_q": safe_q.detach().mean().item(),
            "safe_target": safe_target.detach().mean().item(),
            "collected_rollouts": collected_rollouts,
        }
