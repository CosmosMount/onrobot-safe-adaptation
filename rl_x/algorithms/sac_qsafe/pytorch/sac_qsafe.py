import os
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.algorithms.sac.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.sac.pytorch.policy import get_policy
from rl_x.algorithms.sac.pytorch.critic import get_critic
from rl_x.algorithms.sac.pytorch.entropy_coefficient import get_entropy_coefficient
from rl_x.algorithms.sac.pytorch.replay_buffer import ReplayBuffer
from rl_x.algorithms.qsafe.common import (
    CompletedTrajectoryCollector,
    extract_failure_signal,
    restore_algorithm_config,
    validate_safety_rollout_environment,
)
from rl_x.algorithms.qsafe.pytorch import QSafe

rlx_logger = logging.getLogger("rl_x")


class SAC_QSafe:
    def __init__(
        self, config, train_env, eval_env, run_path, writer, _defer_transfer_load=False
    ):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.compile_mode = config.algorithm.compile_mode
        self.bf16_mixed_precision_training = config.algorithm.bf16_mixed_precision_training
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        validate_safety_rollout_environment(train_env, eval_env, self.nr_envs)
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        if (
            config.algorithm.phase == "pretrain"
            and int(config.algorithm.qsafe.buffer_size) >= int(self.buffer_size)
        ):
            raise ValueError(
                "SQRL requires algorithm.qsafe.buffer_size to be smaller than "
                "algorithm.buffer_size so D_safe remains a recent on-policy replay."
            )
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.target_entropy = config.algorithm.target_entropy
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.phase = config.algorithm.phase
        if self.phase not in ("pretrain", "finetune"):
            raise ValueError("algorithm.phase must be 'pretrain' or 'finetune'.")
        self.n_off = int(config.algorithm.n_off)
        self.n_safe = int(config.algorithm.n_safe)
        if self.n_off < 1:
            raise ValueError("algorithm.n_off must be at least 1.")
        if self.phase == "pretrain" and self.n_safe < 1:
            raise ValueError("algorithm.n_safe must be at least 1 during pretraining.")
        self.qsafe_updates_per_iteration = int(
            config.algorithm.qsafe.updates_per_iteration
        )
        if self.qsafe_updates_per_iteration < 1:
            raise ValueError("algorithm.qsafe.updates_per_iteration must be at least 1.")

        if config.algorithm.device == "gpu" and torch.cuda.is_available():
            device_name = "cuda"
        elif config.algorithm.device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_name = "mps"
        else:
            device_name = "cpu"
        self.device = torch.device(device_name)
        rlx_logger.info(f"Using device: {self.device}")

        if self.bf16_mixed_precision_training and self.device.type != "cuda":
            raise ValueError("bfloat16 mixed precision training is only supported on CUDA devices.")

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high

        self.policy = get_policy(config, self.train_env, self.device)
        self.critic = get_critic(config, self.train_env, self.device)
        self.entropy_coefficient = get_entropy_coefficient(config, self.train_env, self.device)
        self.qsafe = QSafe(
            config,
            self.train_env,
            self.device,
            self.rng,
            self.phase,
            defer_checkpoint_load=_defer_transfer_load,
        )

        if self.phase == "finetune" and not _defer_transfer_load:
            self._load_pretrained_policy(str(config.algorithm.pretrained_policy_path))
        
        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, fused=fused)
        self.q_optimizer = optim.Adam(list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()), lr=self.learning_rate, fused=fused)
        self.entropy_optimizer = optim.Adam([self.entropy_coefficient.log_alpha], lr=self.learning_rate, fused=fused)
        self.nu = torch.tensor(
            float(config.algorithm.initial_nu),
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.phase == "finetune",
        )
        self.dual_optimizer = None
        if self.phase == "finetune":
            self.dual_optimizer = optim.Adam(
                [self.nu], lr=float(config.algorithm.dual_learning_rate), fused=fused
            )

        if self.anneal_learning_rate:
            self.q_scheduler = optim.lr_scheduler.LinearLR(self.q_optimizer, start_factor=1.0, end_factor=0.0, total_iters=(self.total_timesteps - self.learning_starts) // self.nr_envs)
            self.policy_scheduler = optim.lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.0, total_iters=(self.total_timesteps - self.learning_starts) // self.nr_envs)
            self.entropy_scheduler = optim.lr_scheduler.LinearLR(self.entropy_optimizer, start_factor=1.0, end_factor=0.0, total_iters=(self.total_timesteps - self.learning_starts) // self.nr_envs)

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.best_mean_return = -np.inf

    def _load_pretrained_policy(self, file_path):
        if not file_path:
            raise ValueError("algorithm.pretrained_policy_path is required for finetune.")
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        if "policy_state_dict" in checkpoint:
            policy_state_dict = checkpoint["policy_state_dict"]
        elif "state_dict" in checkpoint:
            policy_state_dict = checkpoint["state_dict"]
        else:
            policy_state_dict = checkpoint
        self.policy.load_state_dict(policy_state_dict)

    def _sample_unconstrained_action(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return self.policy.get_action(states)[:2]

    def _sample_policy_candidates(self, states, phase=None):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        nr_candidates = self.qsafe.candidate_actions
        candidate_states = states[:, None, :].expand(-1, nr_candidates, -1)
        flat_states = candidate_states.reshape(self.nr_envs * nr_candidates, -1)
        candidate_actions, candidate_processed_actions, candidate_log_probs = self.policy.get_action(flat_states)
        candidate_actions = candidate_actions.reshape(
            self.nr_envs, nr_candidates, *self.train_env.single_action_space.shape
        )
        candidate_processed_actions = candidate_processed_actions.reshape_as(candidate_actions)
        candidate_log_probs = candidate_log_probs.reshape(self.nr_envs, nr_candidates)
        selected_actions, selected_indices, metrics = self.qsafe.select_safe_action(
            states, candidate_actions, candidate_log_probs, phase or self.phase
        )
        batch_indices = torch.arange(self.nr_envs, device=self.device)
        selected_processed_actions = candidate_processed_actions[
            batch_indices, selected_indices
        ]
        return selected_actions, selected_processed_actions, metrics

    
    def train(self):
        @torch.compile(mode=self.compile_mode)
        def policy_and_entropy_loss_fn(batch_states):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                current_actions, _, current_log_probs = self.policy.get_action(batch_states)

                q1 = self.critic.q1(batch_states, current_actions)
                q2 = self.critic.q2(batch_states, current_actions)

                min_q = torch.minimum(q1, q2)

                alpha = self.entropy_coefficient()
                alpha_detach = alpha.detach()
                policy_loss = (alpha_detach * current_log_probs - min_q).mean()
                safety_q = torch.zeros_like(min_q)
                if self.phase == "finetune":
                    # QSafe parameters are frozen, but this forward pass intentionally
                    # remains differentiable with respect to the sampled action.
                    safety_q = self.qsafe.values(batch_states, current_actions)
                    policy_loss = policy_loss + self.nu.detach() * (
                        safety_q - self.qsafe.epsilon
                    ).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()

            policy_grad_norm = 0.0
            for param in self.policy.parameters():
                policy_grad_norm += param.grad.detach().data.norm(2) ** 2
            policy_grad_norm = policy_grad_norm ** 0.5

            self.policy_optimizer.step()

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                entropy_detach = -current_log_probs.detach()
                entropy_detach_mean = entropy_detach.mean()
                entropy_loss = self.entropy_coefficient.loss(entropy_detach).mean()

            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()

            entropy_grad_norm = self.entropy_coefficient.log_alpha.grad.detach().data.norm(2) ** 2
            
            self.entropy_optimizer.step()

            dual_loss = torch.zeros((), dtype=torch.float32, device=self.device)
            if self.phase == "finetune":
                dual_loss = self.nu * (self.qsafe.epsilon - safety_q.detach()).mean()
                self.dual_optimizer.zero_grad()
                dual_loss.backward()
                self.dual_optimizer.step()
                with torch.no_grad():
                    self.nu.clamp_(min=0.0)

            return (
                policy_loss,
                entropy_loss,
                min_q,
                entropy_detach_mean,
                alpha_detach,
                policy_grad_norm,
                entropy_grad_norm,
                safety_q.detach().mean(),
                dual_loss.detach(),
                self.nu.detach(),
            )
        

        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(states, next_states, actions, rewards, dones):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    next_actions, _, next_log_probs = self.policy.get_action(next_states)
                    next_q1_target = self.critic.q1_target(next_states, next_actions)
                    next_q2_target = self.critic.q2_target(next_states, next_actions)
                    min_next_q_target = torch.minimum(next_q1_target, next_q2_target)
                    alpha = self.entropy_coefficient().detach()
                    y = rewards.reshape(-1, 1) + self.gamma * (1 - dones.reshape(-1, 1)) * (min_next_q_target - alpha * next_log_probs)

                q1 = self.critic.q1(states, actions)
                q2 = self.critic.q2(states, actions)
                q1_loss = F.mse_loss(q1, y)
                q2_loss = F.mse_loss(q2, y)
                q_loss = (q1_loss + q2_loss) / 2
            
            self.q_optimizer.zero_grad()
            q_loss.backward()

            q1_grad_norm = 0.0
            q2_grad_norm = 0.0
            for param in self.critic.q1.parameters():
                q1_grad_norm += param.grad.detach().data.norm(2) ** 2
            for param in self.critic.q2.parameters():
                q2_grad_norm += param.grad.detach().data.norm(2) ** 2
            critic_grad_norm = q1_grad_norm ** 0.5 + q2_grad_norm ** 0.5

            self.q_optimizer.step()

            return q_loss, critic_grad_norm


        self.set_train_mode()

        offline_replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.rng, self.device)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        safety_state = None
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        nr_failures = 0
        nr_safe_env_steps = 0
        nr_safe_rollouts = 0
        nr_safe_failures = 0
        pretrain_stage = "task"
        task_steps_this_iteration = 0
        safety_collector = None
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        safety_metrics_collection = {}
        evaluation_metrics_collection = {}
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None
        
        while global_step < self.total_timesteps or (
            self.phase == "pretrain" and pretrain_stage == "safe"
        ):
            start_time = time.time()
            torch.compiler.cudagraph_mark_step_begin()
            if logging_time_prev:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)


            # Algorithm 1 alternates n_off unconstrained task steps with n_safe
            # complete on-policy rollouts from the QSafe-projected policy.
            is_safety_step = self.phase == "pretrain" and pretrain_stage == "safe"
            if is_safety_step and safety_state is None:
                safety_state, _ = self.eval_env.reset()
            acting_state = safety_state if is_safety_step else state
            interaction_env = self.eval_env if is_safety_step else self.train_env
            completed_safety_block = False
            dones_this_rollout = 0
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                if is_safety_step:
                    action, processed_action, action_safety_metrics = self._sample_policy_candidates(
                        acting_state, phase="pretrain"
                    )
                elif self.phase == "finetune":
                    action, processed_action, action_safety_metrics = self._sample_policy_candidates(
                        acting_state, phase="finetune"
                    )
                else:
                    action, processed_action = self._sample_unconstrained_action(acting_state)
                    action_safety_metrics = None
            action = action.cpu().numpy()
            processed_action = processed_action.cpu().numpy()
            if action_safety_metrics is not None:
                for key, value in action_safety_metrics.items():
                    safety_metrics_collection.setdefault(key, []).append(value)
            
            next_state, reward, terminated, truncated, info = interaction_env.step(
                processed_action
            )
            failure = extract_failure_signal(info, terminated, self.nr_envs)
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = np.array(
                        interaction_env.get_final_observation_at_index(info, i)
                    )
                    if not is_safety_step:
                        saving_return_buffer.append(
                            interaction_env.get_final_info_value_at_index(
                                info, "episode_return", i
                            )
                        )
                    dones_this_rollout += 1
            if not is_safety_step:
                for key, info_value in self.train_env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(key, []).extend(info_value)

            if is_safety_step:
                completed_trajectories = safety_collector.add_step(
                    acting_state,
                    actual_next_state,
                    action,
                    failure,
                    terminated,
                    truncated,
                )
                for trajectory in completed_trajectories:
                    self.qsafe.add_trajectory(trajectory)
                    nr_safe_rollouts += 1
                nr_safe_env_steps += self.nr_envs
                nr_safe_failures += int(np.sum(failure))
                safety_metrics_collection.setdefault("qsafe/safe_rollout_failure_rate", []).append(
                    float(np.mean(failure))
                )
                if safety_collector.complete:
                    completed_safety_block = True
                    safety_metrics_collection.setdefault(
                        "qsafe/replay_transitions", []
                    ).append(self.qsafe.replay_buffer.nr_transitions)
                    safety_metrics_collection.setdefault(
                        "qsafe/replay_trajectories", []
                    ).append(self.qsafe.replay_buffer.nr_trajectories)
                    pretrain_stage = "task"
                    task_steps_this_iteration = 0
                    safety_collector = None
                    safety_state = None
                else:
                    safety_state = next_state
            else:
                # D_offline is exclusively populated by task-policy interaction.
                offline_replay_buffer.add(state, actual_next_state, action, reward, terminated)
                global_step += self.nr_envs
                nr_episodes += dones_this_rollout
                nr_failures += int(np.sum(failure))
                safety_metrics_collection.setdefault("qsafe/task_failure_rate", []).append(
                    float(np.mean(failure))
                )
                task_steps_this_iteration += 1
                state = next_state
                if self.phase == "pretrain" and (
                    task_steps_this_iteration >= self.n_off
                    or global_step >= self.total_timesteps
                ):
                    pretrain_stage = "safe"
                    safety_collector = CompletedTrajectoryCollector(
                        self.nr_envs, self.n_safe
                    )
                    safety_state = None

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = (
                not is_safety_step
                and global_step > self.learning_starts
                and offline_replay_buffer.size > 0
            )
            should_optimize = should_learning_start
            should_evaluate = (
                not is_safety_step
                and self.evaluation_frequency != -1
                and global_step % self.evaluation_frequency == 0
            )
            should_try_to_save = (
                not is_safety_step
                and should_learning_start
                and self.save_model
                and dones_this_rollout > 0
            )
            should_log = (
                not is_safety_step and global_step % self.logging_frequency == 0
            )

            
            # Optimizing - Prepare batches
            if should_optimize:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = offline_replay_buffer.sample(self.batch_size)


            # Optimizing - Q-functions, policy and entropy coefficient
            if should_optimize:
                # Critic loss
                q_loss, critic_grad_norm = critic_loss_fn(batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations)

                # Update critic targets
                with torch.no_grad():
                    for param, target_param in zip(self.critic.q1.parameters(), self.critic.q1_target.parameters()):
                        target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
                    for param, target_param in zip(self.critic.q2.parameters(), self.critic.q2_target.parameters()):
                        target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

                # Policy and entropy loss
                (
                    policy_loss,
                    entropy_loss,
                    min_q,
                    entropy,
                    alpha,
                    policy_grad_norm,
                    entropy_grad_norm,
                    safety_q,
                    dual_loss,
                    nu,
                ) = policy_and_entropy_loss_fn(batch_states)

                # Create metrics
                optimization_metrics = {
                    "entropy/alpha": alpha.item(),
                    "entropy/entropy": entropy.item(),
                    "gradients/policy_grad_norm": policy_grad_norm.item(),
                    "gradients/critic_grad_norm": critic_grad_norm.item(),
                    "gradients/entropy_grad_norm": entropy_grad_norm.item(),
                    "loss/q_loss": q_loss.item(),
                    "loss/policy_loss": policy_loss.item(),
                    "loss/entropy_loss": entropy_loss.item(),
                    "lr/learning_rate": self.learning_rate if not self.anneal_learning_rate else self.q_scheduler.get_last_lr()[0],
                    "q_value/q_value": min_q.mean().item(),
                    "qsafe/actor_value": safety_q.item(),
                    "qsafe/nu": nu.item(),
                    "loss/qsafe_dual_loss": dual_loss.item(),
                }

                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_updates += 1
            
                if self.anneal_learning_rate:
                    self.q_scheduler.step()
                    self.policy_scheduler.step()
                    self.entropy_scheduler.step()

            if completed_safety_block and self.qsafe.ready_to_update():
                def sample_unconstrained_action(next_states):
                    with torch.no_grad():
                        next_actions, _, _ = self.policy.get_action(next_states)
                    return next_actions

                for _ in range(self.qsafe_updates_per_iteration):
                    qsafe_metrics = self.qsafe.update(sample_unconstrained_action)
                    for key, value in qsafe_metrics.items():
                        safety_metrics_collection.setdefault(key, []).append(value)
            
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                while True:
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        _, eval_processed_action, _ = self._sample_policy_candidates(eval_state)
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(eval_processed_action.cpu().numpy())
                    eval_failure = extract_failure_signal(
                        eval_info, eval_terminated, self.nr_envs
                    )
                    evaluation_metrics_collection.setdefault(
                        "eval/failure_rate", []
                    ).extend(eval_failure.tolist())
                    eval_done = eval_terminated | eval_truncated
                    for i, single_done in enumerate(eval_done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics_collection.setdefault("eval/episode_return", []).append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_return", i))
                            evaluation_metrics_collection.setdefault("eval/episode_length", []).append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                self.set_train_mode()
            
            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)


            # Logging
            if should_log:
                self.start_logging(global_step)

                steps_metrics["steps/nr_env_steps"] = global_step
                steps_metrics["steps/nr_task_env_steps"] = global_step
                steps_metrics["steps/nr_safe_env_steps"] = nr_safe_env_steps
                steps_metrics["steps/nr_safe_rollouts"] = nr_safe_rollouts
                steps_metrics["steps/nr_updates"] = nr_updates
                steps_metrics["steps/nr_episodes"] = nr_episodes
                steps_metrics["steps/nr_failures"] = nr_failures
                steps_metrics["steps/nr_safe_failures"] = nr_safe_failures

                rollout_info_metrics = {}
                env_info_metrics = {}
                if step_info_collection:
                    info_names = list(step_info_collection.keys())
                    for info_name in info_names:
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = np.mean(step_info_collection[info_name])
                        if mean_value == mean_value:  # Check if mean_value is NaN
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value
                
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
                safety_metrics = {
                    key: np.mean(value) for key, value in safety_metrics_collection.items()
                }
                combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics, **safety_metrics}
                for key, value in combined_metrics.items():
                    self.log(f"{key}", value, global_step)

                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                safety_metrics_collection = {}
                evaluation_metrics_collection = {}

                self.end_logging()
            
            logging_end_time = time.time()
            logging_time_prev = logging_end_time - saving_end_time

        if self.save_model:
            self.save("final.model")


    def log(self, name, value, step):
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self, wandb_commit=True):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache, commit=wandb_commit)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self, model_file_name="best.model"):
        file_path = os.path.join(self.save_path, model_file_name)
        save_dict = {
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "q1_state_dict": self.critic.q1.state_dict(),
            "q2_state_dict": self.critic.q2.state_dict(),
            "q1_target_state_dict": self.critic.q1_target.state_dict(),
            "q2_target_state_dict": self.critic.q2_target.state_dict(),
            "log_alpha": self.entropy_coefficient.log_alpha,
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "entropy_optimizer_state_dict": self.entropy_optimizer.state_dict(),
            "qsafe_state_dict": self.qsafe.state_dict(
                include_optimizer=self.phase == "pretrain"
            ),
            "nu": self.nu.detach().cpu(),
        }
        if self.dual_optimizer is not None:
            save_dict["dual_optimizer_state_dict"] = self.dual_optimizer.state_dict()
        torch.save(save_dict, file_path)
        torch.save(
            {"policy_state_dict": self.policy.state_dict()},
            self.save_path + "/policy.model",
        )
        torch.save(
            {
                "q1_state_dict": self.critic.q1.state_dict(),
                "q2_state_dict": self.critic.q2.state_dict(),
                "q1_target_state_dict": self.critic.q1_target.state_dict(),
                "q2_target_state_dict": self.critic.q2_target.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            },
            self.save_path + "/task_critic.model",
        )
        self.qsafe.save(
            self.save_path + "/qsafe.model",
            include_optimizer=self.phase == "pretrain",
        )
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))
    

    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(
            config.runner.load_model, map_location="cpu", weights_only=False
        )
        loaded_algorithm_config = checkpoint["config_algorithm"]
        restore_algorithm_config(
            config.algorithm,
            loaded_algorithm_config,
            explicitly_set_algorithm_params,
        )
        model = SAC_QSafe(
            config, train_env, eval_env, run_path, writer, _defer_transfer_load=True
        )
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.q1.load_state_dict(checkpoint["q1_state_dict"])
        model.critic.q2.load_state_dict(checkpoint["q2_state_dict"])
        model.critic.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        model.critic.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        model.entropy_coefficient.log_alpha = checkpoint["log_alpha"]
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        model.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer_state_dict"])
        model.qsafe.load_state_dict(
            checkpoint["qsafe_state_dict"], load_optimizer=model.phase == "pretrain"
        )
        if model.phase == "finetune":
            model.qsafe.freeze()
        model.nu.data.copy_(checkpoint["nu"].to(model.device))
        if model.dual_optimizer is not None and "dual_optimizer_state_dict" in checkpoint:
            model.dual_optimizer.load_state_dict(checkpoint["dual_optimizer_state_dict"])
        return model

    
    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    _, processed_action, _ = self._sample_policy_candidates(state)
                state, reward, terminated, truncated, info = self.eval_env.step(processed_action.cpu().numpy())
                extract_failure_signal(info, terminated, self.nr_envs)
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        self.policy.train()
        self.critic.q1.train()
        self.critic.q2.train()
        self.critic.q1_target.train()
        self.critic.q2_target.train()
        if not self.qsafe.frozen:
            self.qsafe.online.train()
            self.qsafe.target.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.q1.eval()
        self.critic.q2.eval()
        self.critic.q1_target.eval()
        self.critic.q2_target.eval()
        self.qsafe.online.eval()
        self.qsafe.target.eval()

    
    def general_properties():
        return GeneralProperties
