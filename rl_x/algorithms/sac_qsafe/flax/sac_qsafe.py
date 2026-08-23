import os
import logging
import time
from collections import deque
import tree
import numpy as np
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import flax
from flax import serialization
from flax.training.train_state import TrainState
import optax
import wandb

from rl_x.algorithms.sac.flax.general_properties import GeneralProperties
from rl_x.algorithms.sac.flax.policy import get_policy
from rl_x.algorithms.sac.flax.critic import get_critic
from rl_x.algorithms.sac.flax.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.sac.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.sac.flax.rl_train_state import RLTrainState
from rl_x.algorithms.qsafe.common import (
    CompletedTrajectoryCollector,
    extract_failure_signal,
    restore_algorithm_config,
    validate_safety_rollout_environment,
)
from rl_x.algorithms.qsafe.flax import QSafe

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
        self.nu = float(config.algorithm.initial_nu)
        self.dual_learning_rate = float(config.algorithm.dual_learning_rate)
        self.dual_optimizer = (
            optax.adam(self.dual_learning_rate)
            if self.phase == "finetune"
            else optax.set_to_zero()
        )
        self.dual_optimizer_state = self.dual_optimizer.init(
            jnp.asarray(self.nu, dtype=jnp.float32)
        )

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, entropy_coefficient_key = jax.random.split(self.key, 4)

        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)
        
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.train_env.single_action_space.shape).item()
        else:
            self.target_entropy = float(self.target_entropy)
        self.entropy_coefficient = EntropyCoefficient(1.0)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.entropy_coefficient.apply = jax.jit(self.entropy_coefficient.apply)

        def linear_schedule(count):
            step = (count * self.nr_envs) - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction
        
        self.q_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.policy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.entropy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([self.train_env.single_observation_space.sample()])
        action = jnp.array([self.train_env.single_action_space.sample()])

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.policy_learning_rate)
        )

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state, action),
            target_params=self.critic.init(critic_key, state, action),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.q_learning_rate)
        )

        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.entropy_learning_rate)
        )

        self.key, qsafe_key = jax.random.split(self.key)
        self.qsafe = QSafe(
            config,
            self.train_env,
            self.rng,
            qsafe_key,
            self.phase,
            defer_checkpoint_load=_defer_transfer_load,
        )
        if self.phase == "finetune" and not _defer_transfer_load:
            self._load_pretrained_policy(str(config.algorithm.pretrained_policy_path))

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"

    def _load_pretrained_policy(self, file_path):
        if not file_path:
            raise ValueError("algorithm.pretrained_policy_path is required for finetune.")
        with open(file_path, "rb") as policy_file:
            params = serialization.from_bytes(self.policy_state.params, policy_file.read())
        self.policy_state = self.policy_state.replace(params=params)

    def _sample_unconstrained_action(self, states):
        states = jnp.asarray(states, dtype=jnp.float32)
        action_mean, action_logstd = self.policy.apply(self.policy_state.params, states)
        self.key, action_key = jax.random.split(self.key)
        actions = jnp.tanh(
            action_mean
            + jnp.exp(action_logstd)
            * jax.random.normal(action_key, shape=action_mean.shape)
        )
        return actions, self.get_processed_action(actions)

    def _sample_policy_candidates(self, states, phase=None):
        states = jnp.asarray(states, dtype=jnp.float32)
        nr_candidates = self.qsafe.candidate_actions
        action_mean, action_logstd = self.policy.apply(self.policy_state.params, states)
        action_mean = jnp.repeat(action_mean[:, None, :], nr_candidates, axis=1)
        action_logstd = jnp.repeat(action_logstd[:, None, :], nr_candidates, axis=1)
        self.key, action_key = jax.random.split(self.key)
        action_std = jnp.exp(action_logstd)
        pretanh = action_mean + action_std * jax.random.normal(
            action_key, shape=action_mean.shape
        )
        candidate_actions = jnp.tanh(pretanh)
        log_probs = (
            -0.5 * ((pretanh - action_mean) / action_std) ** 2
            - 0.5 * jnp.log(2.0 * jnp.pi)
            - action_logstd
        )
        log_probs -= jnp.log(1.0 - candidate_actions ** 2 + 1e-6)
        log_probs = jnp.sum(log_probs, axis=-1)
        selected_actions, _, metrics = self.qsafe.select_safe_action(
            states, candidate_actions, log_probs, phase or self.phase
        )
        return selected_actions, self.get_processed_action(selected_actions), metrics
        
    
    def train(self):
        @jax.jit
        def update(
                policy_state: TrainState, critic_state: RLTrainState, entropy_coefficient_state: TrainState,
                states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminations: np.ndarray,
                qsafe_params: flax.core.FrozenDict, nu: float, key: jax.random.PRNGKey
            ):
            def loss_fn(policy_params: flax.core.FrozenDict, critic_params: flax.core.FrozenDict, entropy_coefficient_params: flax.core.FrozenDict,
                        state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: np.ndarray, terminated: np.ndarray,
                        key1: jax.random.PRNGKey, key2: jax.random.PRNGKey
                ):
                # Critic loss
                next_action_mean, next_action_logstd = self.policy.apply(stop_gradient(policy_params), next_state)
                next_action_std = jnp.exp(next_action_logstd)
                next_action_pretanh = next_action_mean + next_action_std * jax.random.normal(key1, shape=next_action_mean.shape)
                next_action = jnp.tanh(next_action_pretanh)
                next_log_prob = -0.5 * ((next_action_pretanh - next_action_mean) / next_action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - next_action_logstd
                next_log_prob -= jnp.log(1.0 - next_action ** 2 + 1e-6)
                next_log_prob = jnp.sum(next_log_prob, axis=-1)

                alpha_with_grad = self.entropy_coefficient.apply(entropy_coefficient_params)
                alpha = stop_gradient(alpha_with_grad)

                next_q_target = self.critic.apply(critic_state.target_params, next_state, next_action)
                min_next_q_target = jnp.min(next_q_target)

                y = reward + self.gamma * (1 - terminated) * (min_next_q_target - alpha * next_log_prob)

                q = self.critic.apply(critic_params, state, action)
                q_loss = (q - y) ** 2

                # Policy loss
                current_action_mean, current_action_logstd = self.policy.apply(policy_params, state)
                current_action_std = jnp.exp(current_action_logstd)
                current_action_pretanh = current_action_mean + current_action_std * jax.random.normal(key2, shape=current_action_mean.shape)
                current_action = jnp.tanh(current_action_pretanh)
                current_log_prob = -0.5 * ((current_action_pretanh - current_action_mean) / current_action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - current_action_logstd
                current_log_prob -= jnp.log(1.0 - current_action ** 2 + 1e-6)
                current_log_prob = jnp.sum(current_log_prob, axis=-1)
                entropy = stop_gradient(-current_log_prob)

                q = self.critic.apply(stop_gradient(critic_params), state, current_action)
                min_q = jnp.min(q)

                policy_loss = alpha * current_log_prob - min_q
                safety_q = jnp.zeros_like(min_q)
                if self.phase == "finetune":
                    safety_q = self.qsafe.network.apply(
                        stop_gradient(qsafe_params), state, current_action
                    ).squeeze()
                    policy_loss = policy_loss + nu * (
                        safety_q - self.qsafe.epsilon
                    )

                # Entropy loss
                entropy_loss = alpha_with_grad * (entropy - self.target_entropy)

                # Combine losses
                loss = q_loss + policy_loss + entropy_loss

                # Create metrics
                metrics = {
                    "loss/q_loss": q_loss,
                    "loss/policy_loss": policy_loss,
                    "loss/entropy_loss": entropy_loss,
                    "entropy/entropy": entropy,
                    "entropy/alpha": alpha,
                    "q_value/q_value": min_q,
                    "qsafe/actor_value": safety_q,
                }

                return loss, (metrics)
            

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1, 2), has_aux=True)

            keys = jax.random.split(key, (self.batch_size * 2) + 1)
            key, keys1, keys2 = keys[0], keys[1::2], keys[2::2]

            (loss, (metrics)), (policy_gradients, critic_gradients, entropy_gradients) = grad_loss_fn(
                policy_state.params, critic_state.params, entropy_coefficient_state.params,
                states, next_states, actions, rewards, terminations, keys1, keys2)

            policy_state = policy_state.apply_gradients(grads=policy_gradients)
            critic_state = critic_state.apply_gradients(grads=critic_gradients)
            entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_gradients)

            # Update targets
            critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau))

            metrics["lr/learning_rate"] = policy_state.opt_state.hyperparams["learning_rate"]
            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
            metrics["gradients/entropy_grad_norm"] = optax.global_norm(entropy_gradients)

            return policy_state, critic_state, entropy_coefficient_state, metrics, key
        

        self.set_train_mode()

        offline_replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.rng)

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
            if logging_time_prev:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)


            is_safety_step = self.phase == "pretrain" and pretrain_stage == "safe"
            if is_safety_step and safety_state is None:
                safety_state, _ = self.eval_env.reset()
            acting_state = safety_state if is_safety_step else state
            interaction_env = self.eval_env if is_safety_step else self.train_env
            completed_safety_block = False
            dones_this_rollout = 0
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
            if action_safety_metrics is not None:
                for key, value in action_safety_metrics.items():
                    safety_metrics_collection.setdefault(key, []).append(value)
            
            next_state, reward, terminated, truncated, info = interaction_env.step(
                jax.device_get(processed_action)
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

            host_action = jax.device_get(action)
            if is_safety_step:
                completed_trajectories = safety_collector.add_step(
                    acting_state,
                    actual_next_state,
                    host_action,
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
                offline_replay_buffer.add(state, actual_next_state, host_action, reward, terminated)
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
                self.policy_state, self.critic_state, self.entropy_coefficient_state, optimization_metrics, self.key = update(
                    self.policy_state,
                    self.critic_state,
                    self.entropy_coefficient_state,
                    batch_states,
                    batch_next_states,
                    batch_actions,
                    batch_rewards,
                    batch_terminations,
                    self.qsafe.state.params,
                    self.nu,
                    self.key,
                )
                if self.phase == "finetune":
                    safety_value = float(optimization_metrics["qsafe/actor_value"])
                    nu_before_update = jnp.asarray(self.nu, dtype=jnp.float32)
                    dual_gradient = jnp.asarray(
                        self.qsafe.epsilon - safety_value, dtype=jnp.float32
                    )
                    dual_updates, self.dual_optimizer_state = (
                        self.dual_optimizer.update(
                            dual_gradient,
                            self.dual_optimizer_state,
                            nu_before_update,
                        )
                    )
                    updated_nu = optax.apply_updates(
                        nu_before_update, dual_updates
                    )
                    self.nu = max(0.0, float(updated_nu))
                    optimization_metrics["qsafe/nu"] = self.nu
                    optimization_metrics["loss/qsafe_dual_loss"] = float(
                        nu_before_update * dual_gradient
                    )
                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_updates += 1

            if completed_safety_block and self.qsafe.ready_to_update():
                def sample_unconstrained_action(next_states, action_key):
                    action_mean, action_logstd = self.policy.apply(
                        self.policy_state.params, next_states
                    )
                    action_std = jnp.exp(action_logstd)
                    return jnp.tanh(
                        action_mean
                        + action_std
                        * jax.random.normal(action_key, shape=action_mean.shape)
                    )

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
                    _, eval_processed_action, _ = self._sample_policy_candidates(eval_state)
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(jax.device_get(eval_processed_action))
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
        checkpoint = {
            "policy": self.policy_state,
            "critic": self.critic_state,
            "entropy_coefficient": self.entropy_coefficient_state,
            "qsafe": self.qsafe.state,
            "nu": jnp.asarray(self.nu, dtype=jnp.float32),
            "dual_optimizer_state": self.dual_optimizer_state,
        }
        payload = {
            "config_algorithm": self.config.algorithm.to_dict(),
            "checkpoint": serialization.to_state_dict(checkpoint),
            "qsafe_metadata": self.qsafe.metadata(),
        }
        model_file_path = os.path.join(self.save_path, model_file_name)
        with open(model_file_path, "wb") as model_file:
            model_file.write(serialization.msgpack_serialize(payload))

        with open(f"{self.save_path}/policy.msgpack", "wb") as policy_file:
            policy_file.write(serialization.to_bytes(self.policy_state.params))
        with open(f"{self.save_path}/task_critic.msgpack", "wb") as critic_file:
            critic_file.write(serialization.to_bytes(self.critic_state))
        self.qsafe.save(
            f"{self.save_path}/qsafe.msgpack",
            include_optimizer=self.phase == "pretrain",
        )

        if self.track_wandb:
            wandb.save(model_file_path, base_path=self.save_path)


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        with open(config.runner.load_model, "rb") as model_file:
            payload = serialization.msgpack_restore(model_file.read())
        loaded_algorithm_config = payload["config_algorithm"]
        restore_algorithm_config(
            config.algorithm,
            loaded_algorithm_config,
            explicitly_set_algorithm_params,
        )
        model = SAC_QSafe(
            config, train_env, eval_env, run_path, writer, _defer_transfer_load=True
        )
        if "qsafe_metadata" not in payload:
            raise ValueError(
                "The Flax SAC-QSafe checkpoint is missing QSafe metadata and "
                "cannot be safely resumed."
            )
        model.qsafe._validate_metadata(payload["qsafe_metadata"])

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "entropy_coefficient": model.entropy_coefficient_state,
            "qsafe": model.qsafe.state,
            "nu": jnp.asarray(model.nu, dtype=jnp.float32),
            "dual_optimizer_state": model.dual_optimizer_state,
        }
        checkpoint = serialization.from_state_dict(target, payload["checkpoint"])

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.entropy_coefficient_state = checkpoint["entropy_coefficient"]
        model.qsafe.state = checkpoint["qsafe"]
        if model.phase == "finetune":
            model.qsafe.freeze()
        model.nu = float(checkpoint["nu"])
        model.dual_optimizer_state = checkpoint["dual_optimizer_state"]

        return model
    

    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                _, processed_action, _ = self._sample_policy_candidates(state)
                state, reward, terminated, truncated, info = self.eval_env.step(jax.device_get(processed_action))
                extract_failure_signal(info, terminated, self.nr_envs)
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
