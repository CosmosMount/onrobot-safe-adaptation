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

from rl_x.algorithms.sac_qsafe.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.sac_qsafe.pytorch.observation_normalizer import (
    ObservationNormalizer,
)
from rl_x.algorithms.sac_qsafe.pytorch.checkpoint import restore_parameter_
from rl_x.algorithms.sac_qsafe.pytorch.rollout import (
    AtomicTrajectoryUpdateBudget,
    TransitionUpdateBudget,
    preserve_policy_outputs,
)
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
from rl_x.environments.safety_rollout import InvalidTransitionError

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
        self.qsafe_enabled = bool(config.algorithm.qsafe.enabled)
        if self.phase == "pretrain" and not self.qsafe_enabled:
            raise ValueError("QSafe cannot be disabled during SQRL pre-training.")
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
        self.task_utd_ratio = float(config.algorithm.task_utd_ratio)
        # Validate the ratio at construction rather than after simulator startup.
        TransitionUpdateBudget(self.task_utd_ratio)

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
        self._env_as_low_tensor = torch.as_tensor(
            self.env_as_low, dtype=torch.float32, device=self.device
        )
        self._env_as_high_tensor = torch.as_tensor(
            self.env_as_high, dtype=torch.float32, device=self.device
        )

        self.observation_normalizer = ObservationNormalizer(
            self.train_env.single_observation_space.shape[0],
            enabled=bool(config.algorithm.enable_observation_normalization),
            epsilon=float(config.algorithm.normalizer_epsilon),
        ).to(self.device)

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
            self._load_pretrained_task_critic(
                str(config.algorithm.pretrained_task_critic_path)
            )
            self.observation_normalizer.freeze()
        
        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, fused=fused)
        self.q_optimizer = optim.Adam(list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()), lr=self.learning_rate, fused=fused)
        self.entropy_optimizer = optim.Adam([self.entropy_coefficient.log_alpha], lr=self.learning_rate, fused=fused)
        self.nu = torch.tensor(
            float(config.algorithm.initial_nu) if self.qsafe_enabled else 0.0,
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.phase == "finetune" and self.qsafe_enabled,
        )
        self.dual_optimizer = None
        if self.phase == "finetune" and self.qsafe_enabled:
            self.dual_optimizer = optim.Adam(
                [self.nu], lr=float(config.algorithm.dual_learning_rate), fused=fused
            )
        if self.phase == "finetune":
            rlx_logger.info(
                "QSafe fine-tuning constraints: "
                + (
                    "enabled"
                    if self.qsafe_enabled
                    else "disabled (SAC ablation)"
                )
            )

        if self.anneal_learning_rate:
            if str(config.algorithm.rollout_mode) == "partitioned":
                scheduler_updates = int(
                    max(
                        1,
                        (self.total_timesteps - self.learning_starts)
                        * self.task_utd_ratio,
                    )
                )
            else:
                scheduler_updates = int(
                    max(
                        1,
                        (self.total_timesteps - self.learning_starts)
                        // self.nr_envs,
                    )
                )
            self.q_scheduler = optim.lr_scheduler.LinearLR(
                self.q_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=scheduler_updates,
            )
            self.policy_scheduler = optim.lr_scheduler.LinearLR(
                self.policy_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=scheduler_updates,
            )
            self.entropy_scheduler = optim.lr_scheduler.LinearLR(
                self.entropy_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=scheduler_updates,
            )

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
        if "log_alpha" in checkpoint:
            restore_parameter_(
                self.entropy_coefficient.log_alpha, checkpoint["log_alpha"]
            )
        else:
            rlx_logger.warning(
                "Pretrained policy artifact has no SAC entropy temperature; "
                f"using algorithm.alpha_init={float(self.config.algorithm.alpha_init):g}"
            )

        normalizer_state = checkpoint.get("observation_normalizer_state_dict")
        if normalizer_state is None:
            raise ValueError(
                "Pretrained policy checkpoint is missing observation normalizer state."
            )
        metadata = checkpoint.get("observation_normalizer_metadata")
        if metadata is not None:
            self.observation_normalizer.validate_metadata(metadata)
        self.observation_normalizer.load_state_dict(normalizer_state)
        environment_manifest = checkpoint.get("environment_manifest")
        if hasattr(self.train_env, "validate_transfer_checkpoint_manifest"):
            if environment_manifest is None:
                raise ValueError("Pretrained policy is missing environment manifest.")
            self.train_env.validate_transfer_checkpoint_manifest(
                environment_manifest, self.observation_normalizer.metadata()
            )
        else:
            raise ValueError(
                "Fine-tuning environment must validate the policy transfer manifest"
            )

    def _load_pretrained_task_critic(self, file_path):
        if not file_path:
            raise ValueError(
                "algorithm.pretrained_task_critic_path is required for finetune."
            )
        checkpoint = torch.load(
            file_path, map_location=self.device, weights_only=False
        )
        required = {
            "q1_state_dict",
            "q2_state_dict",
            "q1_target_state_dict",
            "q2_target_state_dict",
        }
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(
                f"Pretrained task critic checkpoint is missing {sorted(missing)}"
            )
        self.critic.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.critic.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.critic.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.critic.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        rlx_logger.info("Transferred pretrained SAC task critics and targets")

    def _normalize_states(self, states, update=False):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return self.observation_normalizer.normalize(states, update=update)

    def _project_actions(self, raw_states, actions):
        """Apply an optional environment-provided, differentiable action map.

        The hook operates on normalized actions and raw observations and must
        return a tensor with the same shape.  Keeping it optional preserves the
        generic SAC-QSafe interface, while environments with state-dependent
        clipping or rate limits can make replay, actor, target, and QSafe action
        semantics identical.
        """

        project_actions = getattr(self.train_env, "project_actions", None)
        if project_actions is None:
            return actions
        projected = project_actions(raw_states, actions)
        if not isinstance(projected, torch.Tensor):
            projected = torch.as_tensor(
                projected, dtype=actions.dtype, device=actions.device
            )
        else:
            projected = projected.to(dtype=actions.dtype, device=actions.device)
        if projected.shape != actions.shape:
            raise ValueError(
                "environment.project_actions(raw_states, raw_actions) must "
                f"preserve action shape: {projected.shape} != {actions.shape}"
            )
        if actions.requires_grad and not projected.requires_grad:
            raise ValueError(
                "environment.project_actions must be differentiable when used "
                "by the actor update"
            )
        return projected

    def _process_normalized_actions(self, actions):
        actions = actions.clamp(-1.0, 1.0)
        return self._env_as_low_tensor + 0.5 * (actions + 1.0) * (
            self._env_as_high_tensor - self._env_as_low_tensor
        )

    def _sample_unconstrained_action(self, states, update_normalizer=False):
        raw_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        normalized_states = self._normalize_states(
            raw_states, update=update_normalizer
        )
        actions, _, _ = self.policy.get_action(normalized_states)
        actions = self._project_actions(raw_states, actions)
        return actions, self._process_normalized_actions(actions)

    def _sample_policy_candidates(
        self, states, phase=None, update_normalizer=False
    ):
        raw_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        states = self._normalize_states(raw_states, update=update_normalizer)
        nr_envs = states.shape[0]
        nr_candidates = self.qsafe.candidate_actions
        candidate_states = states[:, None, :].expand(-1, nr_candidates, -1)
        flat_states = candidate_states.reshape(nr_envs * nr_candidates, -1)
        candidate_actions, _, candidate_log_probs = self.policy.get_action(flat_states)
        candidate_actions = candidate_actions.reshape(
            nr_envs, nr_candidates, *self.train_env.single_action_space.shape
        )
        candidate_raw_states = raw_states[:, None, :].expand(
            -1, nr_candidates, -1
        )
        candidate_actions = self._project_actions(
            candidate_raw_states, candidate_actions
        )
        candidate_processed_actions = self._process_normalized_actions(
            candidate_actions
        )
        candidate_log_probs = candidate_log_probs.reshape(nr_envs, nr_candidates)
        selected_actions, selected_indices, metrics = self.qsafe.select_safe_action(
            states, candidate_actions, candidate_log_probs, phase or self.phase
        )
        batch_indices = torch.arange(nr_envs, device=self.device)
        selected_processed_actions = candidate_processed_actions[
            batch_indices, selected_indices
        ]
        return selected_actions, selected_processed_actions, metrics

    
    def train(self):
        if str(self.config.algorithm.rollout_mode) == "partitioned":
            return self._train_partitioned()

        @torch.compile(mode=self.compile_mode)
        def policy_and_entropy_loss_fn(batch_states, raw_batch_states):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                current_actions, _, current_log_probs = self.policy.get_action(batch_states)
                current_actions = self._project_actions(
                    raw_batch_states, current_actions
                )

                q1 = self.critic.q1(batch_states, current_actions)
                q2 = self.critic.q2(batch_states, current_actions)

                min_q = torch.minimum(q1, q2)

                alpha = self.entropy_coefficient()
                alpha_detach = alpha.detach()
                policy_loss = (alpha_detach * current_log_probs - min_q).mean()
                safety_q = torch.zeros_like(min_q)
                if self.phase == "finetune" and self.qsafe_enabled:
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
            if self.phase == "finetune" and self.qsafe_enabled:
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
        def critic_loss_fn(
            states, next_states, actions, rewards, dones, raw_next_states
        ):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    next_actions, _, next_log_probs = self.policy.get_action(next_states)
                    next_actions = self._project_actions(
                        raw_next_states, next_actions
                    )
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
                elif self.phase == "finetune" and self.qsafe_enabled:
                    action, processed_action, action_safety_metrics = self._sample_policy_candidates(
                        acting_state, phase="finetune"
                    )
                else:
                    action, processed_action = self._sample_unconstrained_action(
                        acting_state, update_normalizer=True
                    )
                    action_safety_metrics = None
            action = action.cpu().numpy()
            processed_action = processed_action.cpu().numpy()
            if action_safety_metrics is not None:
                for key, value in action_safety_metrics.items():
                    safety_metrics_collection.setdefault(key, []).append(value)
            
            try:
                next_state, reward, terminated, truncated, info = interaction_env.step(
                    processed_action
                )
            except InvalidTransitionError as exc:
                rlx_logger.warning(
                    "Discarding invalid environment transition: %s", exc
                )
                recovered_state, _ = interaction_env.reset()
                if is_safety_step:
                    safety_state = recovered_state
                    safety_collector = CompletedTrajectoryCollector(
                        self.nr_envs, self.n_safe
                    )
                else:
                    state = recovered_state
                continue
            failure = extract_failure_signal(info, terminated, self.nr_envs)
            applied_action = np.asarray(
                info.get("applied_action", action), dtype=np.float32
            ).reshape(action.shape)
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
                    applied_action,
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
                offline_replay_buffer.add(
                    state,
                    actual_next_state,
                    applied_action,
                    reward,
                    terminated,
                )
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
                normalized_batch_states = self._normalize_states(
                    batch_states, update=False
                )
                normalized_batch_next_states = self._normalize_states(
                    batch_next_states, update=False
                )


            # Optimizing - Q-functions, policy and entropy coefficient
            if should_optimize:
                # Critic loss
                q_loss, critic_grad_norm = critic_loss_fn(
                    normalized_batch_states,
                    normalized_batch_next_states,
                    batch_actions,
                    batch_rewards,
                    batch_terminations,
                    batch_next_states,
                )

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
                ) = policy_and_entropy_loss_fn(
                    normalized_batch_states, batch_states
                )

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
                    qsafe_metrics = self.qsafe.update(
                        sample_unconstrained_action,
                        state_transform=lambda value: self._normalize_states(
                            value, update=False
                        ),
                        action_transform=self._project_actions,
                    )
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
                        if self.qsafe_enabled:
                            _, eval_processed_action, _ = self._sample_policy_candidates(
                                eval_state
                            )
                        else:
                            raw_eval_state = torch.as_tensor(
                                eval_state, dtype=torch.float32, device=self.device
                            )
                            normalized_eval_state = self._normalize_states(
                                raw_eval_state, update=False
                            )
                            eval_action = self.policy.get_deterministic_action(
                                normalized_eval_state
                            )
                            eval_action = self._project_actions(
                                raw_eval_state, eval_action
                            )
                            eval_processed_action = self._process_normalized_actions(
                                eval_action
                            )
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

    def _partitioned_task_update(self, replay_buffer):
        torch.compiler.cudagraph_mark_step_begin()
        raw_states, raw_next_states, actions, rewards, terminations = replay_buffer.sample(
            self.batch_size
        )
        states = self._normalize_states(raw_states, update=False)
        next_states = self._normalize_states(raw_next_states, update=False)

        with autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.bf16_mixed_precision_training,
        ):
            with torch.no_grad():
                next_actions, _, next_log_probs = self.policy.get_action(next_states)
                next_actions = self._project_actions(
                    raw_next_states, next_actions
                )
                next_q = torch.minimum(
                    self.critic.q1_target(next_states, next_actions),
                    self.critic.q2_target(next_states, next_actions),
                )
                target = rewards[:, None] + self.gamma * (
                    1.0 - terminations[:, None]
                ) * (
                    next_q
                    - self.entropy_coefficient().detach() * next_log_probs
                )
            q1 = self.critic.q1(states, actions)
            q2 = self.critic.q2(states, actions)
            q_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            for online, target_network in (
                (self.critic.q1, self.critic.q1_target),
                (self.critic.q2, self.critic.q2_target),
            ):
                for parameter, target_parameter in zip(
                    online.parameters(), target_network.parameters()
                ):
                    target_parameter.data.mul_(1.0 - self.tau).add_(
                        parameter.data, alpha=self.tau
                    )

        with autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.bf16_mixed_precision_training,
        ):
            current_actions, _, log_probs = self.policy.get_action(states)
            current_actions = self._project_actions(raw_states, current_actions)
            task_q = torch.minimum(
                self.critic.q1(states, current_actions),
                self.critic.q2(states, current_actions),
            )
            policy_loss = (
                self.entropy_coefficient().detach() * log_probs - task_q
            ).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        entropy_loss = self.entropy_coefficient.loss(-log_probs.detach()).mean()
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()
        if self.anneal_learning_rate:
            self.q_scheduler.step()
            self.policy_scheduler.step()
            self.entropy_scheduler.step()
        return {
            # Keep optimizer metrics on-device. The partitioned loop aggregates
            # them and performs one host synchronization per logging interval,
            # rather than one ``item()`` synchronization per gradient update.
            "loss/q_loss": q_loss.detach(),
            "loss/policy_loss": policy_loss.detach(),
            "loss/entropy_loss": entropy_loss.detach(),
            # Read from the parameter instead of the compiled forward output;
            # CUDA Graph outputs are reused and cannot be retained by the
            # interval accumulator across optimizer calls.
            "entropy/alpha": self.entropy_coefficient.log_alpha.detach().exp(),
            "lr/learning_rate": (
                self.learning_rate
                if not self.anneal_learning_rate
                else self.q_scheduler.get_last_lr()[0]
            ),
        }

    def _train_partitioned(self):
        from rl_x.algorithms.qsafe.common import VectorTrajectoryAccumulator

        if self.phase != "pretrain":
            raise ValueError("partitioned rollout mode is only valid for pretraining")
        if not hasattr(self.train_env, "step_partitions"):
            raise ValueError(
                "partitioned rollout mode requires an environment implementing "
                "step_partitions()"
            )
        nr_task_envs = int(self.train_env.nr_task_envs)
        nr_safety_envs = int(self.train_env.nr_safety_envs)
        task_state, safety_state = self.train_env.reset_partitions()
        task_replay = ReplayBuffer(
            int(self.buffer_size),
            nr_task_envs,
            self.train_env.single_observation_space.shape,
            self.train_env.single_action_space.shape,
            self.rng,
            self.device,
        )
        safety_trajectories = VectorTrajectoryAccumulator(nr_safety_envs)
        global_step = 0
        safety_step_count = 0
        vector_step_count = 0
        task_episode_count = 0
        safety_episode_count = 0
        task_failure_count = 0
        safety_failure_count = 0
        training_start_time = time.perf_counter()
        interval_start_time = training_start_time
        previous_log_step = 0
        previous_log_task_updates = 0
        previous_log_qsafe_updates = 0
        task_update_budget = TransitionUpdateBudget(self.task_utd_ratio)
        qsafe_update_budget = AtomicTrajectoryUpdateBudget(
            self.qsafe_updates_per_iteration
        )
        interval_metric_sums = {}
        interval_metric_counts = {}

        def record_interval_metrics(metrics):
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().float()
                else:
                    value = float(value)
                if name in interval_metric_sums:
                    interval_metric_sums[name] = interval_metric_sums[name] + value
                else:
                    interval_metric_sums[name] = value
                interval_metric_counts[name] = interval_metric_counts.get(name, 0) + 1
        logging_frequency = int(self.logging_frequency)
        if logging_frequency < 1:
            raise ValueError("algorithm.logging_frequency must be at least 1")
        next_log_step = int(self.logging_frequency)
        checkpoint_frequency = int(self.config.algorithm.checkpoint_frequency)
        next_checkpoint = checkpoint_frequency
        self.set_train_mode()

        while global_step < self.total_timesteps:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.bf16_mixed_precision_training,
            ):
                task_action, task_processed = self._sample_unconstrained_action(
                    task_state, update_normalizer=True
                )
                # The safety call replays the same compiled policy and may
                # overwrite its CUDA Graph output buffers.  Keep task outputs
                # in independent storage until both partitions are stepped.
                task_action, task_processed = preserve_policy_outputs(
                    task_action, task_processed
                )
                safety_action, safety_processed, projection_metrics = (
                    self._sample_policy_candidates(safety_state, phase="pretrain")
                )
            task_action = task_action.cpu().numpy()
            safety_action = safety_action.cpu().numpy()
            try:
                task_step, safety_step = self.train_env.step_partitions(
                    task_processed.cpu().numpy(), safety_processed.cpu().numpy()
                )
            except InvalidTransitionError as exc:
                rlx_logger.warning(
                    "Discarding invalid partitioned transition: %s", exc
                )
                task_state, safety_state = self.train_env.reset_partitions()
                safety_trajectories = VectorTrajectoryAccumulator(nr_safety_envs)
                continue

            task_failure = extract_failure_signal(
                task_step.info, task_step.terminated, nr_task_envs
            )
            safety_failure = extract_failure_signal(
                safety_step.info, safety_step.terminated, nr_safety_envs
            )
            task_done = task_step.terminated | task_step.truncated
            safety_done = safety_step.terminated | safety_step.truncated
            task_next = np.asarray(task_step.observation).copy()
            safety_next = np.asarray(safety_step.observation).copy()
            for index in np.flatnonzero(task_done):
                final = task_step.info["final_observation"][index]
                if final is not None:
                    task_next[index] = final
            for index in np.flatnonzero(safety_done):
                final = safety_step.info["final_observation"][index]
                if final is not None:
                    safety_next[index] = final

            task_applied = np.asarray(
                task_step.info.get("applied_action", task_action), dtype=np.float32
            )
            safety_applied = np.asarray(
                safety_step.info.get("applied_action", safety_action),
                dtype=np.float32,
            )
            task_replay.add(
                task_state,
                task_next,
                task_applied,
                task_step.reward,
                task_step.terminated,
            )
            completed = safety_trajectories.add_step(
                safety_state,
                safety_next,
                safety_applied,
                safety_failure,
                safety_step.terminated,
                safety_step.truncated,
            )
            for trajectory in completed:
                self.qsafe.add_trajectory(trajectory)

            previous_global_step = global_step
            global_step += nr_task_envs
            safety_step_count += nr_safety_envs
            vector_step_count += 1
            task_episode_count += int(np.count_nonzero(task_done))
            safety_episode_count += len(completed)
            task_failure_count += int(np.sum(task_failure))
            safety_failure_count += int(np.sum(safety_failure))
            record_interval_metrics(
                {
                    "rollout/task_step_reward": np.mean(task_step.reward),
                    "rollout/safety_step_reward": np.mean(safety_step.reward),
                    "failures/task_rate": np.mean(task_failure),
                    "failures/safety_rate": np.mean(safety_failure),
                }
            )
            for pool_name, rollout_step in (
                ("task", task_step),
                ("safety", safety_step),
            ):
                for metric_name, values in self.train_env.get_logging_info_dict(
                    rollout_step.info
                ).items():
                    values = np.asarray(values, dtype=np.float32)
                    if values.size:
                        record_interval_metrics(
                            {
                                f"env/{pool_name}/{metric_name}": np.mean(values)
                            }
                        )
            for index in np.flatnonzero(task_done):
                final_info = task_step.info["final_info"][index]
                if final_info is not None:
                    record_interval_metrics(
                        {
                            "rollout/task_episode_return": final_info[
                                "episode_return"
                            ],
                            "rollout/task_episode_length": final_info[
                                "episode_length"
                            ],
                        }
                    )
            for index in np.flatnonzero(safety_done):
                final_info = safety_step.info["final_info"][index]
                if final_info is not None:
                    record_interval_metrics(
                        {
                            "rollout/safety_episode_return": final_info[
                                "episode_return"
                            ],
                            "rollout/safety_episode_length": final_info[
                                "episode_length"
                            ],
                        }
                    )
            eligible_before = max(
                0, previous_global_step - int(self.learning_starts)
            )
            eligible_after = max(0, global_step - int(self.learning_starts))
            task_update_budget.add_transitions(eligible_after - eligible_before)
            qsafe_update_budget.add_completed(completed)
            task_state = task_step.observation
            safety_state = safety_step.observation

            if task_replay.size > 0:
                for _ in range(task_update_budget.consume_ready_updates()):
                    task_metrics = self._partitioned_task_update(task_replay)
                    record_interval_metrics(task_metrics)

            if self.qsafe.ready_to_update():
                def sample_unconstrained_action(normalized_states):
                    with torch.no_grad():
                        return self.policy.get_action(normalized_states)[0]

                for _ in range(qsafe_update_budget.consume_ready_updates()):
                    torch.compiler.cudagraph_mark_step_begin()
                    qsafe_metrics = self.qsafe.update(
                        sample_unconstrained_action,
                        state_transform=lambda value: self._normalize_states(
                            value, update=False
                        ),
                        action_transform=self._project_actions,
                    )
                    record_interval_metrics(qsafe_metrics)
            record_interval_metrics(projection_metrics)

            if global_step >= next_log_step or global_step >= self.total_timesteps:
                now = time.perf_counter()
                interval_seconds = max(now - interval_start_time, 1e-12)
                self.start_logging(global_step)
                self.log(
                    "steps/nr_vector_steps", vector_step_count, global_step
                )
                self.log("steps/nr_task_env_steps", global_step, global_step)
                self.log(
                    "steps/nr_safe_env_steps", safety_step_count, global_step
                )
                self.log(
                    "steps/nr_task_updates", task_update_budget.updates, global_step
                )
                self.log(
                    "steps/nr_qsafe_updates", qsafe_update_budget.updates, global_step
                )
                self.log(
                    "episodes/nr_task_completed", task_episode_count, global_step
                )
                self.log(
                    "episodes/nr_safety_completed", safety_episode_count, global_step
                )
                self.log(
                    "failures/nr_task", task_failure_count, global_step
                )
                self.log(
                    "failures/nr_safety", safety_failure_count, global_step
                )
                self.log(
                    "replay/task_transitions",
                    task_replay.size * nr_task_envs,
                    global_step,
                )
                self.log(
                    "replay/qsafe_transitions",
                    self.qsafe.replay_buffer.nr_transitions,
                    global_step,
                )
                self.log(
                    "replay/qsafe_trajectories",
                    self.qsafe.replay_buffer.nr_trajectories,
                    global_step,
                )
                self.log(
                    "replay/qsafe_committed_transitions",
                    qsafe_update_budget.transitions,
                    global_step,
                )
                self.log("utd/task_configured", self.task_utd_ratio, global_step)
                self.log(
                    "utd/qsafe_updates_per_trajectory_configured",
                    self.qsafe_updates_per_iteration,
                    global_step,
                )
                self.log(
                    "utd/task_effective",
                    task_update_budget.effective_ratio,
                    global_step,
                )
                self.log(
                    "utd/qsafe_updates_per_trajectory_effective",
                    qsafe_update_budget.effective_ratio,
                    global_step,
                )
                self.log("utd/task_credit", task_update_budget.credit, global_step)
                self.log("utd/qsafe_credit", qsafe_update_budget.credit, global_step)
                self.log(
                    "time/task_transitions_per_second",
                    (global_step - previous_log_step) / interval_seconds,
                    global_step,
                )
                self.log(
                    "time/task_updates_per_second",
                    (task_update_budget.updates - previous_log_task_updates)
                    / interval_seconds,
                    global_step,
                )
                self.log(
                    "time/qsafe_updates_per_second",
                    (qsafe_update_budget.updates - previous_log_qsafe_updates)
                    / interval_seconds,
                    global_step,
                )
                self.log(
                    "time/elapsed_seconds", now - training_start_time, global_step
                )
                for name, value_sum in interval_metric_sums.items():
                    mean_value = value_sum / interval_metric_counts[name]
                    if isinstance(mean_value, torch.Tensor):
                        mean_value = mean_value.item()
                    self.log(name, float(mean_value), global_step)
                self.end_logging()
                interval_metric_sums = {}
                interval_metric_counts = {}
                interval_start_time = now
                previous_log_step = global_step
                previous_log_task_updates = task_update_budget.updates
                previous_log_qsafe_updates = qsafe_update_budget.updates
                while next_log_step <= global_step:
                    next_log_step += logging_frequency
            if (
                self.save_model
                and checkpoint_frequency > 0
                and global_step >= next_checkpoint
            ):
                self.save(f"step_{global_step:09d}.model")
                while next_checkpoint <= global_step:
                    next_checkpoint += checkpoint_frequency

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
        environment_manifest = None
        if hasattr(self.train_env, "checkpoint_manifest"):
            environment_manifest = self.train_env.checkpoint_manifest(
                self.observation_normalizer.metadata()
            )
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
            "observation_normalizer_state_dict": self.observation_normalizer.state_dict(),
            "observation_normalizer_metadata": self.observation_normalizer.metadata(),
            "environment_manifest": environment_manifest,
            "nu": self.nu.detach().cpu(),
        }
        if self.dual_optimizer is not None:
            save_dict["dual_optimizer_state_dict"] = self.dual_optimizer.state_dict()
        torch.save(save_dict, file_path)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "log_alpha": self.entropy_coefficient.log_alpha.detach().cpu(),
                "observation_normalizer_state_dict": self.observation_normalizer.state_dict(),
                "observation_normalizer_metadata": self.observation_normalizer.metadata(),
                "environment_manifest": environment_manifest,
            },
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
        # Preserve the Parameter object already referenced by entropy_optimizer.
        # Replacing it would leave the optimizer updating a stale parameter and,
        # because the checkpoint is initially mapped to CPU, can also introduce
        # a device mismatch on GPU resumes.
        restore_parameter_(
            model.entropy_coefficient.log_alpha, checkpoint["log_alpha"]
        )
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        model.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer_state_dict"])
        if "observation_normalizer_state_dict" not in checkpoint:
            raise ValueError("Checkpoint is missing observation normalizer state.")
        if "observation_normalizer_metadata" in checkpoint:
            model.observation_normalizer.validate_metadata(
                checkpoint["observation_normalizer_metadata"]
            )
        model.observation_normalizer.load_state_dict(
            checkpoint["observation_normalizer_state_dict"]
        )
        if hasattr(train_env, "validate_checkpoint_manifest"):
            if checkpoint.get("environment_manifest") is None:
                raise ValueError("Checkpoint is missing environment manifest.")
            train_env.validate_checkpoint_manifest(
                checkpoint["environment_manifest"],
                model.observation_normalizer.metadata(),
            )
        model.qsafe.load_state_dict(
            checkpoint["qsafe_state_dict"], load_optimizer=model.phase == "pretrain"
        )
        if model.phase == "finetune":
            model.qsafe.freeze()
            model.observation_normalizer.freeze()
        model.nu.data.copy_(checkpoint["nu"].to(model.device))
        if model.dual_optimizer is not None and "dual_optimizer_state_dict" in checkpoint:
            model.dual_optimizer.load_state_dict(checkpoint["dual_optimizer_state_dict"])
        return model

    
    def test(self, episodes):
        self.set_eval_mode()
        eval_policy = str(self.config.algorithm.eval_policy)
        if eval_policy not in ("task", "safe"):
            raise ValueError("algorithm.eval_policy must be 'task' or 'safe'")
        if eval_policy == "safe" and not self.qsafe_enabled:
            raise ValueError("Safe evaluation requires algorithm.qsafe.enabled=true")
        for i in range(episodes):
            done = False
            episode_return = 0
            episode_steps = 0
            episode_failures = 0
            forward_velocity_sum = 0.0
            forward_velocity_samples = []
            estimated_forward_velocity_sum = 0.0
            velocity_estimation_error_sum = 0.0
            target_velocity_error_sum = 0.0
            state, _ = self.eval_env.reset()
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    if eval_policy == "task":
                        raw_state = torch.as_tensor(
                            state, dtype=torch.float32, device=self.device
                        )
                        normalized_state = self._normalize_states(
                            raw_state, update=False
                        )
                        action = self.policy.get_deterministic_action(
                            normalized_state
                        )
                        action = self._project_actions(raw_state, action)
                        processed_action = self._process_normalized_actions(action)
                    else:
                        _, processed_action, _ = self._sample_policy_candidates(
                            state
                        )
                state, reward, terminated, truncated, info = self.eval_env.step(processed_action.cpu().numpy())
                failures = extract_failure_signal(info, terminated, self.nr_envs)
                done = terminated | truncated
                episode_return += reward
                episode_steps += 1
                episode_failures += int(np.sum(failures))
                if "forward_velocity" in info:
                    forward_velocity = float(
                        np.mean(np.asarray(info["forward_velocity"]))
                    )
                    forward_velocity_sum += forward_velocity
                    forward_velocity_samples.append(forward_velocity)
                if "estimated_forward_velocity" in info:
                    estimated_forward_velocity_sum += float(
                        np.mean(np.asarray(info["estimated_forward_velocity"]))
                    )
                if "velocity_estimation_error" in info:
                    velocity_estimation_error_sum += float(
                        np.mean(np.asarray(info["velocity_estimation_error"]))
                    )
                if "target_velocity_error" in info:
                    target_velocity_error_sum += float(
                        np.mean(np.asarray(info["target_velocity_error"]))
                    )
            summary = (
                f"Episode {i + 1} - Return: {float(np.mean(episode_return)):.6f}, "
                f"Length: {episode_steps}, Failures: {episode_failures}"
            )
            if episode_steps and "forward_velocity" in info:
                summary += (
                    f", Mean simulator forward velocity: "
                    f"{forward_velocity_sum / episode_steps:.6f}"
                )
                window_size = min(100, len(forward_velocity_samples))
                window_means = [
                    float(np.mean(forward_velocity_samples[start:start + window_size]))
                    for start in range(
                        0,
                        len(forward_velocity_samples) - window_size + 1,
                        window_size,
                    )
                ]
                summary += (
                    f", Last {window_size}-step simulator velocity: "
                    f"{window_means[-1]:.6f}, "
                    f"Min {window_size}-step simulator velocity: "
                    f"{min(window_means):.6f}"
                )
            if episode_steps and "estimated_forward_velocity" in info:
                summary += (
                    f", Mean estimated forward velocity: "
                    f"{estimated_forward_velocity_sum / episode_steps:.6f}"
                )
            if episode_steps and "velocity_estimation_error" in info:
                summary += (
                    f", Mean 3D velocity estimation error: "
                    f"{velocity_estimation_error_sum / episode_steps:.6f}"
                )
            if episode_steps and "target_velocity_error" in info:
                summary += (
                    f", Mean absolute forward target error: "
                    f"{target_velocity_error_sum / episode_steps:.6f}"
                )
            rlx_logger.info(summary)


    def set_train_mode(self):
        self.policy.train()
        self.critic.q1.train()
        self.critic.q2.train()
        self.critic.q1_target.train()
        self.critic.q2_target.train()
        if self.phase == "pretrain" and not self.observation_normalizer.frozen:
            self.observation_normalizer.train()
        if not self.qsafe.frozen:
            self.qsafe.online.train()
            self.qsafe.target.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.q1.eval()
        self.critic.q2.eval()
        self.critic.q1_target.eval()
        self.critic.q2_target.eval()
        self.observation_normalizer.eval()
        self.qsafe.online.eval()
        self.qsafe.target.eval()

    
    def general_properties():
        return GeneralProperties
