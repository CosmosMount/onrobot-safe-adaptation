import logging
import os
import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.amp import autocast

from rl_x.algorithms.crossq.pytorch.critic import get_critic
from rl_x.algorithms.crossq.pytorch.entropy_coefficient import (
    get_entropy_coefficient,
)
from rl_x.algorithms.crossq.pytorch.policy import get_policy
from rl_x.algorithms.crossq.pytorch.replay_buffer import ReplayBuffer
from rl_x.algorithms.crossq_qsafe.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.sac_qsafe.pytorch.observation_normalizer import (
    ObservationNormalizer,
)
from rl_x.algorithms.qsafe.common import (
    CompletedTrajectoryCollector,
    VectorTrajectoryAccumulator,
    extract_failure_signal,
    restore_algorithm_config,
    validate_safety_rollout_environment,
)
from rl_x.algorithms.qsafe.pytorch import QSafe
from rl_x.algorithms.sac_qsafe.pytorch.checkpoint import restore_parameter_
from rl_x.algorithms.sac_qsafe.pytorch.rollout import (
    AtomicTrajectoryUpdateBudget,
    TransitionUpdateBudget,
    preserve_policy_outputs,
)
from rl_x.environments.safety_rollout import InvalidTransitionError


rlx_logger = logging.getLogger("rl_x")


class CrossQ_QSafe:
    """CrossQ task learning with the SQRL safety-policy phases.

    QSafe is deliberately independent from the task critic.  In particular,
    the task critic below keeps CrossQ's defining no-target-network update: the
    current and bootstrapped samples share one training-mode BatchRenorm pass.
    """

    def __init__(
        self,
        config,
        train_env,
        eval_env,
        run_path,
        writer,
        _defer_transfer_load=False,
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
        self.bf16_mixed_precision_training = (
            config.algorithm.bf16_mixed_precision_training
        )
        self.total_timesteps = int(config.algorithm.total_timesteps)
        self.nr_envs = int(config.environment.nr_envs)
        validate_safety_rollout_environment(train_env, eval_env, self.nr_envs)

        self.learning_rate = float(config.algorithm.learning_rate)
        self.anneal_learning_rate = bool(config.algorithm.anneal_learning_rate)
        self.buffer_size = int(config.algorithm.buffer_size)
        self.learning_starts = int(config.algorithm.learning_starts)
        self.batch_size = int(config.algorithm.batch_size)
        self.gamma = float(config.algorithm.gamma)
        self.policy_delay = int(config.algorithm.policy_delay)
        if self.policy_delay < 1:
            raise ValueError("algorithm.policy_delay must be at least 1.")
        self.logging_frequency = int(config.algorithm.logging_frequency)
        if self.logging_frequency < 1:
            raise ValueError("algorithm.logging_frequency must be at least 1.")
        self.evaluation_frequency = int(config.algorithm.evaluation_frequency)
        self.evaluation_episodes = int(config.algorithm.evaluation_episodes)

        self.phase = str(config.algorithm.phase)
        if self.phase not in ("pretrain", "finetune"):
            raise ValueError("algorithm.phase must be 'pretrain' or 'finetune'.")
        self.n_off = int(config.algorithm.n_off)
        self.n_safe = int(config.algorithm.n_safe)
        if self.n_off < 1:
            raise ValueError("algorithm.n_off must be at least 1.")
        if self.phase == "pretrain" and self.n_safe < 1:
            raise ValueError(
                "algorithm.n_safe must be at least 1 during pretraining."
            )
        self.qsafe_updates_per_iteration = int(
            config.algorithm.qsafe.updates_per_iteration
        )
        if self.qsafe_updates_per_iteration < 1:
            raise ValueError(
                "algorithm.qsafe.updates_per_iteration must be at least 1."
            )
        if (
            self.phase == "pretrain"
            and int(config.algorithm.qsafe.buffer_size) >= self.buffer_size
        ):
            raise ValueError(
                "SQRL requires algorithm.qsafe.buffer_size to be smaller than "
                "algorithm.buffer_size so D_safe remains recent and on-policy."
            )
        self.task_utd_ratio = float(config.algorithm.task_utd_ratio)
        TransitionUpdateBudget(self.task_utd_ratio)

        if config.algorithm.device == "gpu" and torch.cuda.is_available():
            device_name = "cuda"
        elif (
            config.algorithm.device == "mps"
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            device_name = "mps"
        else:
            device_name = "cpu"
        self.device = torch.device(device_name)
        rlx_logger.info("Using device: %s", self.device)
        if self.bf16_mixed_precision_training and self.device.type != "cuda":
            raise ValueError(
                "bfloat16 mixed precision training is only supported on CUDA."
            )

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.env_as_low = np.asarray(self.train_env.single_action_space.low)
        self.env_as_high = np.asarray(self.train_env.single_action_space.high)
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
        self.entropy_coefficient = get_entropy_coefficient(
            config, self.train_env, self.device
        )
        self.qsafe = QSafe(
            config,
            self.train_env,
            self.device,
            self.rng,
            self.phase,
            defer_checkpoint_load=_defer_transfer_load,
        )

        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            betas=(float(config.algorithm.policy_adam_b1), 0.999),
            fused=fused,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate,
            betas=(float(config.algorithm.critic_adam_b1), 0.999),
            fused=fused,
        )
        self.entropy_optimizer = optim.Adam(
            [self.entropy_coefficient.log_alpha],
            lr=self.learning_rate,
            fused=fused,
        )
        self.nu = torch.tensor(
            float(config.algorithm.initial_nu),
            dtype=torch.float32,
            device=self.device,
            requires_grad=self.phase == "finetune",
        )
        self.dual_optimizer = None
        if self.phase == "finetune":
            self.dual_optimizer = optim.Adam(
                [self.nu],
                lr=float(config.algorithm.dual_learning_rate),
                fused=fused,
            )

        self.nr_critic_updates = 0
        self.nr_policy_updates = 0
        if self.anneal_learning_rate:
            if str(config.algorithm.rollout_mode) == "partitioned":
                critic_updates = max(
                    1,
                    int(
                        max(0, self.total_timesteps - self.learning_starts)
                        * self.task_utd_ratio
                    ),
                )
            else:
                critic_updates = max(
                    1,
                    (self.total_timesteps - self.learning_starts)
                    // max(1, self.nr_envs),
                )
            policy_updates = max(1, critic_updates // self.policy_delay)
            self.critic_scheduler = optim.lr_scheduler.LinearLR(
                self.critic_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=critic_updates,
            )
            self.policy_scheduler = optim.lr_scheduler.LinearLR(
                self.policy_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=policy_updates,
            )
            self.entropy_scheduler = optim.lr_scheduler.LinearLR(
                self.entropy_optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=policy_updates,
            )

        if self.phase == "finetune" and not _defer_transfer_load:
            self._load_pretrained_policy(
                str(config.algorithm.pretrained_policy_path)
            )
            self.observation_normalizer.freeze()

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.best_mean_return = -np.inf

    def _load_pretrained_policy(self, file_path):
        if not file_path:
            raise ValueError(
                "algorithm.pretrained_policy_path is required for finetune."
            )
        checkpoint = torch.load(
            file_path, map_location=self.device, weights_only=False
        )
        state_dict = checkpoint.get("policy_state_dict", checkpoint)
        self.policy.load_state_dict(state_dict)
        normalizer_state = checkpoint.get("observation_normalizer_state_dict")
        normalizer_metadata = checkpoint.get("observation_normalizer_metadata")
        if normalizer_state is None or normalizer_metadata is None:
            raise ValueError(
                "Pretrained CrossQ-QSafe policy is missing observation "
                "normalizer transfer state."
            )
        self.observation_normalizer.validate_metadata(normalizer_metadata)
        self.observation_normalizer.load_state_dict(normalizer_state)
        if hasattr(self.train_env, "validate_checkpoint_manifest"):
            manifest = checkpoint.get("environment_manifest")
            if manifest is None:
                raise ValueError("Pretrained policy is missing environment manifest.")
            self.train_env.validate_checkpoint_manifest(
                manifest, self.observation_normalizer.metadata()
            )

    def _normalize_states(self, states, update=False):
        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        return self.observation_normalizer.normalize(states, update=update)

    def _project_actions(self, raw_states, actions):
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
                "environment.project_actions must be differentiable in the "
                "actor update"
            )
        return projected

    def _process_normalized_actions(self, actions):
        actions = actions.clamp(-1.0, 1.0)
        return self._env_as_low_tensor + 0.5 * (actions + 1.0) * (
            self._env_as_high_tensor - self._env_as_low_tensor
        )

    def _sample_unconstrained_action(self, states, update_normalizer=False):
        raw_states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        policy_states = self._normalize_states(
            raw_states, update=update_normalizer
        )
        actions, _, _ = self.policy.get_action(policy_states, False)
        actions = self._project_actions(raw_states, actions)
        return actions, self._process_normalized_actions(actions)

    def _sample_policy_candidates(self, states, phase=None):
        raw_states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        policy_states = self._normalize_states(raw_states, update=False)
        nr_envs = raw_states.shape[0]
        nr_candidates = self.qsafe.candidate_actions
        flat_states = (
            policy_states[:, None, :]
            .expand(-1, nr_candidates, -1)
            .reshape(nr_envs * nr_candidates, -1)
        )
        candidate_actions, _, candidate_log_probs = self.policy.get_action(
            flat_states, False
        )
        candidate_actions = candidate_actions.reshape(
            nr_envs,
            nr_candidates,
            *self.train_env.single_action_space.shape,
        )
        repeated_raw_states = raw_states[:, None, :].expand(
            -1, nr_candidates, -1
        )
        candidate_actions = self._project_actions(
            repeated_raw_states, candidate_actions
        )
        candidate_log_probs = candidate_log_probs.reshape(
            nr_envs, nr_candidates
        )
        selected_actions, selected_indices, metrics = (
            self.qsafe.select_safe_action(
                policy_states,
                candidate_actions,
                candidate_log_probs,
                phase or self.phase,
            )
        )
        processed_candidates = self._process_normalized_actions(
            candidate_actions
        )
        batch_indices = torch.arange(nr_envs, device=self.device)
        return (
            selected_actions,
            processed_candidates[batch_indices, selected_indices],
            metrics,
        )

    @staticmethod
    def _replay_size(replay_buffer):
        return (
            replay_buffer.buffer_size
            if replay_buffer.full
            else replay_buffer.pos
        )

    def _critic_update(self, batch):
        raw_states, raw_next_states, actions, rewards, terminations = batch
        states = self._normalize_states(raw_states, update=False)
        next_states = self._normalize_states(raw_next_states, update=False)
        with autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.bf16_mixed_precision_training,
        ):
            with torch.no_grad():
                next_actions, _, next_log_probs = self.policy.get_action(
                    next_states, False
                )
                next_actions = self._project_actions(
                    raw_next_states, next_actions
                )

            # CrossQ invariant: current and next samples share exactly one
            # training-mode critic pass, and there is no target critic.
            current_and_next_q = self.critic(
                torch.cat((states, next_states), dim=0),
                torch.cat((actions, next_actions), dim=0),
                True,
            ).squeeze(-1)
            q_values, next_q_values = torch.split(
                current_and_next_q, states.shape[0], dim=1
            )
            min_next_q = next_q_values.detach().min(dim=0).values
            target = rewards + self.gamma * (1.0 - terminations) * (
                min_next_q
                - self.entropy_coefficient().detach() * next_log_probs
            )
            critic_loss = ((q_values - target) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), float("inf")
        )
        self.critic_optimizer.step()
        self.nr_critic_updates += 1
        if self.anneal_learning_rate:
            self.critic_scheduler.step()
        return {
            "loss/q_loss": critic_loss.detach(),
            "gradients/critic_grad_norm": critic_grad_norm.detach(),
            "lr/critic_learning_rate": self.critic_optimizer.param_groups[0][
                "lr"
            ],
        }

    def _actor_and_entropy_update(self, states, raw_states):
        with autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.bf16_mixed_precision_training,
        ):
            current_actions, _, current_log_probs = self.policy.get_action(
                states, True
            )
            current_actions = self._project_actions(raw_states, current_actions)
            q_values = self.critic(states, current_actions, False).squeeze(-1)
            min_q = q_values.min(dim=0).values
            alpha = self.entropy_coefficient()
            policy_loss = (alpha.detach() * current_log_probs - min_q).mean()
            safety_q = torch.zeros_like(min_q)
            if self.phase == "finetune":
                # QSafe's parameters are frozen, but its action derivative is
                # part of Eq. 4. Squeeze prevents [B] + [B, 1] broadcasting.
                safety_q = self.qsafe.values(states, current_actions).squeeze(-1)
                policy_loss = policy_loss + self.nu.detach() * (
                    safety_q - self.qsafe.epsilon
                ).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), float("inf")
        )
        self.policy_optimizer.step()

        entropy = -current_log_probs.detach()
        entropy_loss = self.entropy_coefficient.loss(entropy).mean()
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        entropy_grad_norm = self.entropy_coefficient.log_alpha.grad.detach().norm(
            2
        )
        self.entropy_optimizer.step()

        dual_loss = torch.zeros((), device=self.device)
        if self.phase == "finetune":
            dual_loss = self.nu * (
                self.qsafe.epsilon - safety_q.detach()
            ).mean()
            self.dual_optimizer.zero_grad()
            dual_loss.backward()
            self.dual_optimizer.step()
            with torch.no_grad():
                self.nu.clamp_(min=0.0)

        self.nr_policy_updates += 1
        if self.anneal_learning_rate:
            self.policy_scheduler.step()
            self.entropy_scheduler.step()
        return {
            "loss/policy_loss": policy_loss.detach(),
            "loss/entropy_loss": entropy_loss.detach(),
            "loss/qsafe_dual_loss": dual_loss.detach(),
            "gradients/policy_grad_norm": policy_grad_norm.detach(),
            "gradients/entropy_grad_norm": entropy_grad_norm.detach(),
            "entropy/entropy": entropy.mean(),
            "entropy/alpha": alpha.detach(),
            "q_value/q_value": min_q.detach().mean(),
            "qsafe/actor_value": safety_q.detach().mean(),
            "qsafe/nu": self.nu.detach(),
            "lr/policy_learning_rate": self.policy_optimizer.param_groups[0][
                "lr"
            ],
        }

    def _task_update(self, replay_buffer):
        batch = replay_buffer.sample(self.batch_size)
        metrics = self._critic_update(batch)
        if self.nr_critic_updates % self.policy_delay == 0:
            states = self._normalize_states(batch[0], update=False)
            metrics.update(self._actor_and_entropy_update(states, batch[0]))
        return metrics

    def _qsafe_update(self):
        def sample_unconstrained_action(next_states):
            with torch.no_grad():
                return self.policy.get_action(next_states, False)[0]

        return self.qsafe.update(
            sample_unconstrained_action,
            state_transform=lambda value: self._normalize_states(
                value, update=False
            ),
            action_transform=self._project_actions,
        )

    @staticmethod
    def _append_metrics(collection, metrics):
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
            collection.setdefault(name, []).append(value)

    @staticmethod
    def _mean_metric(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.stack([value.float() for value in values]).mean().item()
        return float(np.mean(values))

    def train(self):
        if str(self.config.algorithm.rollout_mode) == "partitioned":
            return self._train_partitioned()
        return self._train_serial_reference()

    def _train_serial_reference(self):
        self.set_train_mode()
        offline_replay = ReplayBuffer(
            self.buffer_size,
            self.nr_envs,
            self.train_env.single_observation_space.shape,
            self.train_env.single_action_space.shape,
            self.rng,
            self.device,
        )
        saving_returns = deque(maxlen=100 * self.nr_envs)
        state, _ = self.train_env.reset()
        safety_state = None
        safety_collector = None
        stage = "task"
        task_steps_in_iteration = 0
        global_step = 0
        safe_env_steps = 0
        task_episodes = 0
        safe_episodes = 0
        task_failures = 0
        safe_failures = 0
        metrics = {}
        next_log_step = self.logging_frequency
        next_checkpoint = int(self.config.algorithm.checkpoint_frequency)

        while global_step < self.total_timesteps or (
            self.phase == "pretrain" and stage == "safe"
        ):
            torch.compiler.cudagraph_mark_step_begin()
            is_safety_step = self.phase == "pretrain" and stage == "safe"
            if is_safety_step and safety_state is None:
                safety_state, _ = self.eval_env.reset()
            acting_state = safety_state if is_safety_step else state
            interaction_env = self.eval_env if is_safety_step else self.train_env

            with torch.no_grad(), autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.bf16_mixed_precision_training,
            ):
                if is_safety_step:
                    action, processed_action, projection_metrics = (
                        self._sample_policy_candidates(
                            acting_state, phase="pretrain"
                        )
                    )
                elif self.phase == "finetune":
                    action, processed_action, projection_metrics = (
                        self._sample_policy_candidates(
                            acting_state, phase="finetune"
                        )
                    )
                else:
                    action, processed_action = self._sample_unconstrained_action(
                        acting_state, update_normalizer=True
                    )
                    projection_metrics = {}
            host_action = action.cpu().numpy()
            try:
                next_state, reward, terminated, truncated, info = (
                    interaction_env.step(processed_action.cpu().numpy())
                )
            except InvalidTransitionError as exc:
                rlx_logger.warning(
                    "Discarding invalid environment transition: %s", exc
                )
                recovered, _ = interaction_env.reset()
                if is_safety_step:
                    safety_state = recovered
                    safety_collector = CompletedTrajectoryCollector(
                        self.nr_envs, self.n_safe
                    )
                else:
                    state = recovered
                continue

            failure = extract_failure_signal(info, terminated, self.nr_envs)
            applied_action = np.asarray(
                info.get("applied_action", host_action), dtype=np.float32
            ).reshape(host_action.shape)
            done = terminated | truncated
            actual_next_state = np.asarray(next_state).copy()
            for index in np.flatnonzero(done):
                actual_next_state[index] = np.asarray(
                    interaction_env.get_final_observation_at_index(info, index)
                )
                if not is_safety_step:
                    saving_returns.append(
                        interaction_env.get_final_info_value_at_index(
                            info, "episode_return", index
                        )
                    )

            self._append_metrics(metrics, projection_metrics)
            completed_safety_block = False
            if is_safety_step:
                completed = safety_collector.add_step(
                    acting_state,
                    actual_next_state,
                    applied_action,
                    failure,
                    terminated,
                    truncated,
                )
                for trajectory in completed:
                    self.qsafe.add_trajectory(trajectory)
                safe_env_steps += self.nr_envs
                safe_episodes += len(completed)
                safe_failures += int(np.sum(failure))
                if safety_collector.complete:
                    completed_safety_block = True
                    stage = "task"
                    task_steps_in_iteration = 0
                    safety_collector = None
                    safety_state = None
                else:
                    safety_state = next_state
            else:
                offline_replay.add(
                    state,
                    actual_next_state,
                    applied_action,
                    reward,
                    terminated,
                )
                global_step += self.nr_envs
                task_episodes += int(np.count_nonzero(done))
                task_failures += int(np.sum(failure))
                for name, values in self.train_env.get_logging_info_dict(
                    info
                ).items():
                    metrics.setdefault(name, []).extend(values)
                state = next_state
                task_steps_in_iteration += 1
                if self.phase == "pretrain" and (
                    task_steps_in_iteration >= self.n_off
                    or global_step >= self.total_timesteps
                ):
                    stage = "safe"
                    safety_collector = CompletedTrajectoryCollector(
                        self.nr_envs, self.n_safe
                    )

            if (
                not is_safety_step
                and global_step > self.learning_starts
                and self._replay_size(offline_replay) > 0
            ):
                self._append_metrics(metrics, self._task_update(offline_replay))

            if completed_safety_block and self.qsafe.ready_to_update():
                for _ in range(self.qsafe_updates_per_iteration):
                    self._append_metrics(metrics, self._qsafe_update())

            if (
                not is_safety_step
                and self.evaluation_frequency > 0
                and global_step % self.evaluation_frequency == 0
            ):
                self._append_metrics(metrics, self._evaluate())

            if (
                not is_safety_step
                and self.save_model
                and done.any()
                and saving_returns
            ):
                mean_return = float(np.mean(saving_returns))
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()

            if not is_safety_step and (
                global_step >= next_log_step
                or global_step >= self.total_timesteps
            ):
                self._log_training_metrics(
                    global_step,
                    metrics,
                    safe_env_steps=safe_env_steps,
                    task_episodes=task_episodes,
                    safe_episodes=safe_episodes,
                    task_failures=task_failures,
                    safe_failures=safe_failures,
                )
                metrics = {}
                while next_log_step <= global_step:
                    next_log_step += self.logging_frequency

            checkpoint_frequency = int(
                self.config.algorithm.checkpoint_frequency
            )
            if (
                not is_safety_step
                and self.save_model
                and checkpoint_frequency > 0
                and global_step >= next_checkpoint
            ):
                self.save(f"step_{global_step:09d}.model")
                while next_checkpoint <= global_step:
                    next_checkpoint += checkpoint_frequency

        if self.save_model:
            self.save("final.model")

    def _train_partitioned(self):
        if self.phase != "pretrain":
            raise ValueError(
                "partitioned rollout mode is only valid for pretraining"
            )
        if not hasattr(self.train_env, "step_partitions"):
            raise ValueError(
                "partitioned rollout mode requires step_partitions()"
            )
        nr_task_envs = int(self.train_env.nr_task_envs)
        nr_safety_envs = int(self.train_env.nr_safety_envs)
        task_state, safety_state = self.train_env.reset_partitions()
        task_replay = ReplayBuffer(
            self.buffer_size,
            nr_task_envs,
            self.train_env.single_observation_space.shape,
            self.train_env.single_action_space.shape,
            self.rng,
            self.device,
        )
        safety_trajectories = VectorTrajectoryAccumulator(nr_safety_envs)
        task_budget = TransitionUpdateBudget(self.task_utd_ratio)
        qsafe_budget = AtomicTrajectoryUpdateBudget(
            self.qsafe_updates_per_iteration
        )
        global_step = 0
        safe_env_steps = 0
        task_episodes = 0
        safe_episodes = 0
        task_failures = 0
        safe_failures = 0
        metrics = {}
        next_log_step = self.logging_frequency
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
                task_action, task_processed = preserve_policy_outputs(
                    task_action, task_processed
                )
                safety_action, safety_processed, projection_metrics = (
                    self._sample_policy_candidates(
                        safety_state, phase="pretrain"
                    )
                )
            task_host_action = task_action.cpu().numpy()
            safety_host_action = safety_action.cpu().numpy()
            try:
                task_step, safety_step = self.train_env.step_partitions(
                    task_processed.cpu().numpy(),
                    safety_processed.cpu().numpy(),
                )
            except InvalidTransitionError as exc:
                rlx_logger.warning(
                    "Discarding invalid partitioned transition: %s", exc
                )
                task_state, safety_state = self.train_env.reset_partitions()
                safety_trajectories = VectorTrajectoryAccumulator(
                    nr_safety_envs
                )
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
                task_step.info.get("applied_action", task_host_action),
                dtype=np.float32,
            )
            safety_applied = np.asarray(
                safety_step.info.get("applied_action", safety_host_action),
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

            previous_step = global_step
            global_step += nr_task_envs
            safe_env_steps += nr_safety_envs
            task_episodes += int(np.count_nonzero(task_done))
            safe_episodes += len(completed)
            task_failures += int(np.sum(task_failure))
            safe_failures += int(np.sum(safety_failure))
            eligible_before = max(0, previous_step - self.learning_starts)
            eligible_after = max(0, global_step - self.learning_starts)
            task_budget.add_transitions(eligible_after - eligible_before)
            qsafe_budget.add_completed(completed)
            task_state = task_step.observation
            safety_state = safety_step.observation
            self._append_metrics(metrics, projection_metrics)

            if self._replay_size(task_replay) > 0:
                for _ in range(task_budget.consume_ready_updates()):
                    self._append_metrics(metrics, self._task_update(task_replay))
            if self.qsafe.ready_to_update():
                for _ in range(qsafe_budget.consume_ready_updates()):
                    self._append_metrics(metrics, self._qsafe_update())

            if global_step >= next_log_step or global_step >= self.total_timesteps:
                metrics.setdefault("utd/task_effective", []).append(
                    task_budget.effective_ratio
                )
                metrics.setdefault(
                    "utd/qsafe_updates_per_trajectory_effective", []
                ).append(qsafe_budget.effective_ratio)
                self._log_training_metrics(
                    global_step,
                    metrics,
                    safe_env_steps=safe_env_steps,
                    task_episodes=task_episodes,
                    safe_episodes=safe_episodes,
                    task_failures=task_failures,
                    safe_failures=safe_failures,
                )
                metrics = {}
                while next_log_step <= global_step:
                    next_log_step += self.logging_frequency
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

    def _evaluate(self):
        self.set_eval_mode()
        state, _ = self.eval_env.reset()
        completed = 0
        returns = []
        lengths = []
        failures = []
        while completed < self.evaluation_episodes:
            with torch.no_grad(), autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.bf16_mixed_precision_training,
            ):
                _, processed_action, _ = self._sample_policy_candidates(state)
            state, _, terminated, truncated, info = self.eval_env.step(
                processed_action.cpu().numpy()
            )
            failures.extend(
                extract_failure_signal(info, terminated, self.nr_envs).tolist()
            )
            for index in np.flatnonzero(terminated | truncated):
                returns.append(
                    self.eval_env.get_final_info_value_at_index(
                        info, "episode_return", index
                    )
                )
                lengths.append(
                    self.eval_env.get_final_info_value_at_index(
                        info, "episode_length", index
                    )
                )
                completed += 1
                if completed >= self.evaluation_episodes:
                    break
        self.set_train_mode()
        return {
            "eval/episode_return": float(np.mean(returns)),
            "eval/episode_length": float(np.mean(lengths)),
            "eval/failure_rate": float(np.mean(failures)),
        }

    def _log_training_metrics(
        self,
        global_step,
        metrics,
        *,
        safe_env_steps,
        task_episodes,
        safe_episodes,
        task_failures,
        safe_failures,
    ):
        self.start_logging(global_step)
        fixed = {
            "steps/nr_env_steps": global_step,
            "steps/nr_task_env_steps": global_step,
            "steps/nr_safe_env_steps": safe_env_steps,
            "steps/nr_critic_updates": self.nr_critic_updates,
            "steps/nr_policy_updates": self.nr_policy_updates,
            "episodes/nr_task_completed": task_episodes,
            "episodes/nr_safety_completed": safe_episodes,
            "failures/nr_task": task_failures,
            "failures/nr_safety": safe_failures,
            "replay/qsafe_transitions": self.qsafe.replay_buffer.nr_transitions,
            "replay/qsafe_trajectories": self.qsafe.replay_buffer.nr_trajectories,
        }
        for name, value in fixed.items():
            self.log(name, value, global_step)
        for name, values in metrics.items():
            if values:
                metric_name = (
                    name
                    if "/" in name
                    else (
                        f"rollout/{name}"
                        if name in ("episode_return", "episode_length")
                        else f"env_info/{name}"
                    )
                )
                self.log(
                    metric_name,
                    self._mean_metric(values),
                    global_step,
                )
        self.end_logging()

    def save(self, model_file_name="best.model"):
        os.makedirs(self.save_path, exist_ok=True)
        environment_manifest = None
        if hasattr(self.train_env, "checkpoint_manifest"):
            environment_manifest = self.train_env.checkpoint_manifest(
                self.observation_normalizer.metadata()
            )
        checkpoint = {
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "log_alpha": self.entropy_coefficient.log_alpha.detach().cpu(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "entropy_optimizer_state_dict": self.entropy_optimizer.state_dict(),
            "qsafe_state_dict": self.qsafe.state_dict(
                include_optimizer=self.phase == "pretrain"
            ),
            "observation_normalizer_state_dict": (
                self.observation_normalizer.state_dict()
            ),
            "observation_normalizer_metadata": (
                self.observation_normalizer.metadata()
            ),
            "nu": self.nu.detach().cpu(),
            "nr_critic_updates": self.nr_critic_updates,
            "nr_policy_updates": self.nr_policy_updates,
            "environment_manifest": environment_manifest,
        }
        if self.dual_optimizer is not None:
            checkpoint["dual_optimizer_state_dict"] = (
                self.dual_optimizer.state_dict()
            )
        model_path = os.path.join(self.save_path, model_file_name)
        torch.save(checkpoint, model_path)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "config_algorithm": self.config.algorithm,
                "observation_normalizer_state_dict": (
                    self.observation_normalizer.state_dict()
                ),
                "observation_normalizer_metadata": (
                    self.observation_normalizer.metadata()
                ),
                "environment_manifest": environment_manifest,
            },
            os.path.join(self.save_path, "policy.model"),
        )
        torch.save(
            {
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            os.path.join(self.save_path, "task_critic.model"),
        )
        self.qsafe.save(
            os.path.join(self.save_path, "qsafe.model"),
            include_optimizer=self.phase == "pretrain",
        )
        if self.track_wandb:
            wandb.save(model_path, base_path=self.save_path)

    @staticmethod
    def load(
        config,
        train_env,
        eval_env,
        run_path,
        writer,
        explicitly_set_algorithm_params,
    ):
        checkpoint = torch.load(
            config.runner.load_model, map_location="cpu", weights_only=False
        )
        restore_algorithm_config(
            config.algorithm,
            checkpoint["config_algorithm"],
            explicitly_set_algorithm_params,
        )
        model = CrossQ_QSafe(
            config,
            train_env,
            eval_env,
            run_path,
            writer,
            _defer_transfer_load=True,
        )
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        restore_parameter_(
            model.entropy_coefficient.log_alpha, checkpoint["log_alpha"]
        )
        model.policy_optimizer.load_state_dict(
            checkpoint["policy_optimizer_state_dict"]
        )
        model.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"]
        )
        model.entropy_optimizer.load_state_dict(
            checkpoint["entropy_optimizer_state_dict"]
        )
        model.qsafe.load_state_dict(
            checkpoint["qsafe_state_dict"],
            load_optimizer=model.phase == "pretrain",
        )
        if model.phase == "finetune":
            model.qsafe.freeze()
            model.observation_normalizer.freeze()
        if "observation_normalizer_state_dict" not in checkpoint:
            raise ValueError("Checkpoint is missing observation normalizer state.")
        metadata = checkpoint.get("observation_normalizer_metadata")
        if metadata is None:
            raise ValueError(
                "Checkpoint is missing observation normalizer metadata."
            )
        model.observation_normalizer.validate_metadata(metadata)
        model.observation_normalizer.load_state_dict(
            checkpoint["observation_normalizer_state_dict"]
        )
        with torch.no_grad():
            model.nu.copy_(checkpoint["nu"].to(model.device))
        if (
            model.dual_optimizer is not None
            and "dual_optimizer_state_dict" in checkpoint
        ):
            model.dual_optimizer.load_state_dict(
                checkpoint["dual_optimizer_state_dict"]
            )
        model.nr_critic_updates = int(checkpoint.get("nr_critic_updates", 0))
        model.nr_policy_updates = int(checkpoint.get("nr_policy_updates", 0))
        if hasattr(train_env, "validate_checkpoint_manifest"):
            manifest = checkpoint.get("environment_manifest")
            if manifest is None:
                raise ValueError("Checkpoint is missing environment manifest.")
            train_env.validate_checkpoint_manifest(
                manifest, model.observation_normalizer.metadata()
            )
        return model

    def test(self, episodes):
        original = self.evaluation_episodes
        self.evaluation_episodes = int(episodes)
        result = self._evaluate()
        self.evaluation_episodes = original
        rlx_logger.info(
            "Evaluation over %d episodes - Return: %s, Failure rate: %s",
            episodes,
            result["eval/episode_return"],
            result["eval/failure_rate"],
        )

    def log(self, name, value, step):
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(
            "│ %s│ %s │",
            name.ljust(30),
            str(value).ljust(14)[:14],
            extra={"flush": False},
        )

    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐")
        else:
            rlx_logger.info("Step: %s", step)

    def end_logging(self, wandb_commit=True):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache, commit=wandb_commit)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    def set_train_mode(self):
        self.policy.train()
        self.critic.train()
        if not self.qsafe.frozen:
            self.qsafe.online.train()
            self.qsafe.target.train()
        if not self.observation_normalizer.frozen:
            self.observation_normalizer.train()

    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()
        self.qsafe.online.eval()
        self.qsafe.target.eval()
        self.observation_normalizer.eval()

    @staticmethod
    def general_properties():
        return GeneralProperties
