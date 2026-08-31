import os
import logging
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import optax

from rl_x.algorithms.qsafe.replay_buffer import SafetyReplayBuffer
from rl_x.algorithms.qsafe.common import (
    SafetyObservationHistory,
    VectorTrajectoryAccumulator,
    safety_bellman_target,
    trajectory_with_observation_history,
)
from rl_x.algorithms.qsafe.flax.safety_critic import SafetyQNetwork
from rl_x.algorithms.qsafe.flax.train_state import QSafeTrainState
from rl_x.algorithms.qsafe.flax.projection import resolve_action_projectors


rlx_logger = logging.getLogger("rl_x")


class SafetyObservationNormalizer:
    """Framework-neutral QSafe statistics, independent from actor statistics."""

    def __init__(self, observation_size, enabled=True, epsilon=1e-8):
        self.observation_size = int(observation_size)
        self.enabled = bool(enabled)
        self.epsilon = float(epsilon)
        self.running_mean = np.zeros((1, observation_size), dtype=np.float32)
        self.running_var = np.ones((1, observation_size), dtype=np.float32)
        self.count = 0.0
        self.frozen = False

    def update(self, observations):
        if not self.enabled or self.frozen:
            return
        values = np.asarray(observations, dtype=np.float32).reshape(
            -1, self.observation_size
        )
        if not values.shape[0]:
            return
        batch_count = float(values.shape[0])
        batch_mean = values.mean(axis=0, keepdims=True, dtype=np.float32)
        batch_var = values.var(axis=0, keepdims=True, dtype=np.float32)
        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
            self.count = batch_count
            return
        old_count = np.float32(self.count)
        new_count = old_count + np.float32(batch_count)
        delta = batch_mean - self.running_mean
        mean = self.running_mean + delta * np.float32(batch_count) / new_count
        m2 = (
            self.running_var * old_count
            + batch_var * np.float32(batch_count)
            + np.square(delta) * old_count * np.float32(batch_count) / new_count
        )
        self.running_mean = mean.astype(np.float32)
        self.running_var = (m2 / new_count).astype(np.float32)
        self.count = float(new_count)

    def normalize(self, observations):
        values = jnp.asarray(observations, dtype=jnp.float32)
        if not self.enabled:
            return values
        mean = jnp.asarray(self.running_mean)
        std = jnp.sqrt(jnp.maximum(jnp.asarray(self.running_var), 0.0))
        return (values - mean) / (std + self.epsilon)

    def freeze(self):
        self.frozen = True

    def metadata(self):
        return {
            "observation_size": self.observation_size,
            "enabled": self.enabled,
            "epsilon": self.epsilon,
            "count": int(self.count),
        }

    def state_dict(self):
        return {
            "running_mean": np.asarray(self.running_mean, dtype=np.float32),
            "running_var": np.asarray(self.running_var, dtype=np.float32),
            "count": np.asarray(self.count, dtype=np.float64),
        }

    def load_state_dict(self, state, metadata=None):
        mean = np.asarray(state["running_mean"], dtype=np.float32)
        variance = np.asarray(state["running_var"], dtype=np.float32)
        expected = (1, self.observation_size)
        if mean.shape != expected or variance.shape != expected:
            raise ValueError("Safety normalizer shape mismatch.")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("Safety normalizer contains NaN or infinity.")
        if np.any(variance < 0):
            raise ValueError("Safety normalizer variance must be non-negative.")
        self.running_mean = mean
        self.running_var = variance
        self.count = float(np.asarray(state["count"]).reshape(()))
        if metadata is not None:
            for key in ("observation_size", "enabled", "epsilon"):
                if metadata.get(key) != self.metadata()[key]:
                    raise ValueError(f"Incompatible safety normalizer {key}.")


class QSafe:
    """Flax implementation of the task-critic-independent SQRL layer."""

    def __init__(
        self,
        config,
        env,
        rng,
        key,
        phase=None,
        defer_checkpoint_load=False,
    ):
        self.config = config.algorithm.qsafe
        self.phase = phase or config.algorithm.phase
        self.rng = rng
        self.key = key
        self.version = int(getattr(self.config, "version", 1))
        if self.version not in (1, 2):
            raise ValueError("algorithm.qsafe.version must be 1 or 2.")
        self.checkpoint_version = self.version
        self.epsilon = float(self.config.epsilon)
        self.gamma = float(self.config.gamma)
        self.tau = float(self.config.tau)
        self.batch_size = int(self.config.batch_size)
        self.candidate_actions = int(self.config.candidate_actions)
        if self.candidate_actions < 1:
            raise ValueError("algorithm.qsafe.candidate_actions must be at least 1.")
        self.base_observation_shape = tuple(env.single_observation_space.shape)
        if len(self.base_observation_shape) != 1:
            raise ValueError("QSafe expects a flat policy observation.")
        self.history_length = (
            int(getattr(self.config, "history_length", 1))
            if self.version == 2
            else 1
        )
        self.control_dt = float(getattr(self.config, "control_dt", 0.02))
        if self.history_length < 1 or self.control_dt <= 0:
            raise ValueError("Invalid QSafe history_length or control_dt.")
        self.observation_shape = (
            self.base_observation_shape[0] * self.history_length,
        )
        self.action_shape = tuple(env.single_action_space.shape)
        if self.version == 1:
            self.observation_indices = np.asarray(
                getattr(
                    env,
                    "safety_critic_observation_indices",
                    np.arange(self.observation_shape[0]),
                ),
                dtype=np.int32,
            )
        else:
            self.observation_indices = np.arange(
                self.observation_shape[0], dtype=np.int32
            )
        self.output_activation = "sigmoid" if self.version == 2 else "tanh"
        self.observation_normalizer = SafetyObservationNormalizer(
            self.observation_shape[0],
            enabled=(
                bool(getattr(self.config, "enable_observation_normalization", True))
                if self.version == 2
                else False
            ),
            epsilon=float(getattr(self.config, "normalizer_epsilon", 1e-8)),
        )
        self._rollout_histories = {}
        self.environment_contract = None
        if hasattr(env, "checkpoint_manifest"):
            manifest = env.checkpoint_manifest(None)
            self.environment_contract = {
                "observation": manifest.get("observation"),
                "action": manifest.get("action"),
                "failure": manifest.get("failure"),
            }
        self.calibration_report = {}
        self.network = SafetyQNetwork(
            self.observation_indices.tolist(),
            int(self.config.nr_hidden_units),
            self.output_activation,
        )
        (
            self._jax_project_actions,
            self._host_project_actions,
            self._projector_is_jax,
        ) = resolve_action_projectors(env)
        if self._host_project_actions is not None and not self._projector_is_jax:
            rlx_logger.warning(
                "Environment project_actions is NumPy-only. QSafe rollout actions "
                "and target actions will be projected across a host/device boundary. "
                "Provide a JAX-compatible hook to keep the update path fully JIT."
            )
        self.key, online_key = jax.random.split(self.key)
        dummy_observation = jnp.zeros((1,) + self.observation_shape, dtype=jnp.float32)
        dummy_action = jnp.zeros((1,) + self.action_shape, dtype=jnp.float32)
        params = self.network.init(online_key, dummy_observation, dummy_action)
        # JAX arrays are immutable, so sharing this initial tree value is safe;
        # later TrainState replacements keep online and target trees separate.
        target_params = jax.tree.map(lambda value: value.copy(), params)
        qsafe_optimizer = (
            optax.adam(float(self.config.learning_rate))
            if self.phase == "pretrain" and bool(config.algorithm.qsafe.enabled)
            else optax.set_to_zero()
        )
        self.state = QSafeTrainState.create(
            apply_fn=self.network.apply,
            params=params,
            target_params=target_params,
            tx=qsafe_optimizer,
        )
        self._update_jit = jax.jit(self._update_state)
        self._values_jit = jax.jit(self.network.apply)
        self._select_pretrain_jit = jax.jit(
            lambda params, states, actions, log_probs, key: self._select_kernel(
                params, states, actions, log_probs, key, pretrain=True
            )
        )
        self._select_finetune_jit = jax.jit(
            lambda params, states, actions, log_probs, key: self._select_kernel(
                params,
                states,
                actions,
                log_probs,
                key,
                pretrain=False,
            )
        )
        self.replay_buffer = SafetyReplayBuffer(
            int(self.config.buffer_size),
            config.environment.nr_envs,
            self.observation_shape,
            self.action_shape,
            rng,
            max_trajectories=int(self.config.max_trajectories),
        )
        self.trajectory_accumulator = VectorTrajectoryAccumulator(
            config.environment.nr_envs
        )
        self.frozen = self.phase == "finetune"
        if (
            self.phase == "finetune"
            and bool(config.algorithm.qsafe.enabled)
            and not defer_checkpoint_load
        ):
            checkpoint_path = str(self.config.checkpoint_path)
            if not checkpoint_path:
                raise ValueError("algorithm.qsafe.checkpoint_path is required for finetune.")
            self.load(checkpoint_path, load_optimizer=False)
        if self.phase == "finetune":
            self.freeze()

    def add_transition(self, states, actions, next_states, failures, terminations, truncations):
        if not self.frozen:
            completed = self.trajectory_accumulator.add_step(
                states,
                next_states,
                actions,
                failures,
                terminations,
                truncations,
            )
            for trajectory in completed:
                self.add_trajectory(trajectory)

    def add_trajectory(self, trajectory):
        if not self.frozen:
            if self.version == 2:
                trajectory = trajectory_with_observation_history(
                    trajectory,
                    self.base_observation_shape[0],
                    self.history_length,
                )
                states = np.stack([item[0] for item in trajectory])
                next_states = np.stack([item[1] for item in trajectory])
                self.observation_normalizer.update(
                    np.concatenate([states, next_states], axis=0)
                )
            self.replay_buffer.add_trajectory(trajectory)

    def ready_to_update(self, global_step=None):
        return not self.frozen and self.replay_buffer.nr_transitions > 0

    def _update_state(
        self,
        state,
        states,
        next_states,
        actions,
        failures,
        terminations,
        truncations,
        next_actions,
    ):
        def loss_fn(params):
            predicted = self.network.apply(params, states, actions)
            next_q = self.network.apply(
                jax.lax.stop_gradient(state.target_params), next_states, next_actions
            )
            target = safety_bellman_target(
                failures[:, None],
                terminations[:, None],
                truncations[:, None],
                next_q,
                self.gamma,
            )
            target = jax.lax.stop_gradient(target)
            loss = jnp.mean((predicted - target) ** 2)
            return loss, (predicted, target)

        (loss, (predicted, target)), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        state = state.apply_gradients(grads=gradients)
        state = state.replace(
            target_params=optax.incremental_update(
                state.params, state.target_params, self.tau
            )
        )
        return state, {
            "loss/qsafe_loss": loss,
            "gradients/qsafe_grad_norm": optax.tree.norm(gradients),
            "qsafe/value": jnp.mean(predicted),
            "qsafe/target": jnp.mean(target),
        }

    def update(self, policy_sampler, state_transform=None):
        if self.frozen:
            raise RuntimeError("Frozen QSafe cannot be updated during fine-tuning.")
        raw_states, raw_next_states, actions, failures, terminations, truncations = [
            jnp.asarray(value, dtype=jnp.float32)
            for value in self.replay_buffer.sample(self.batch_size)
        ]
        if self.version == 2:
            states = self.observation_normalizer.normalize(raw_states)
            next_states = self.observation_normalizer.normalize(raw_next_states)
            policy_next_states = raw_next_states[
                ..., -self.base_observation_shape[0] :
            ]
            if state_transform is not None:
                policy_next_states = state_transform(policy_next_states)
        else:
            states = raw_states
            next_states = raw_next_states
            policy_next_states = next_states
            if state_transform is not None:
                states = state_transform(states)
                next_states = state_transform(next_states)
                policy_next_states = next_states
        self.key, action_key = jax.random.split(self.key)
        next_actions = policy_sampler(policy_next_states, action_key)
        if self._projector_is_jax:
            action_raw_states = (
                raw_next_states[..., -self.base_observation_shape[0] :]
                if self.version == 2
                else raw_next_states
            )
            next_actions = self._jax_project_actions(action_raw_states, next_actions)
        elif self._host_project_actions is not None:
            next_actions = jnp.asarray(
                self._host_project_actions(
                    jax.device_get(
                        raw_next_states[..., -self.base_observation_shape[0] :]
                        if self.version == 2
                        else raw_next_states
                    ),
                    jax.device_get(next_actions),
                ),
                dtype=jnp.float32,
            )
        self.state, metrics = self._update_jit(
            self.state,
            states,
            next_states,
            actions,
            failures,
            terminations,
            truncations,
            next_actions,
        )
        return metrics

    def normalize_observations(self, states):
        if self.version == 2:
            return self.observation_normalizer.normalize(states)
        return states

    def values(self, states, actions, normalized=False):
        if self.version == 2 and not normalized:
            states = self.normalize_observations(states)
        return self._values_jit(self.state.params, states, actions)

    def _select_kernel(
        self,
        params,
        states,
        candidate_actions,
        candidate_log_probs,
        selection_key,
        *,
        pretrain,
    ):
        nr_envs, nr_candidates = candidate_actions.shape[:2]
        repeated_states = jnp.repeat(states[:, None, :], nr_candidates, axis=1)
        q_values = self.network.apply(
            params,
            repeated_states.reshape((nr_envs * nr_candidates, -1)),
            candidate_actions.reshape((nr_envs * nr_candidates, -1)),
        ).reshape((nr_envs, nr_candidates))
        safe_mask = q_values < self.epsilon
        fallback = ~jnp.any(safe_mask, axis=1)

        if pretrain:
            scores = jnp.where(safe_mask, q_values, -jnp.inf)
            selected = jnp.argmax(scores, axis=1)
        else:
            logits = candidate_log_probs.reshape((nr_envs, nr_candidates))
            logits = jnp.where(safe_mask, logits, -jnp.inf)
            logits = jnp.where(fallback[:, None], jnp.zeros_like(logits), logits)
            selected = jax.random.categorical(selection_key, logits, axis=-1)

        lowest_risk = jnp.argmin(q_values, axis=1)
        selected = jnp.where(fallback, lowest_risk, selected)
        batch_indices = jnp.arange(nr_envs)
        log_probs = candidate_log_probs.reshape((nr_envs, nr_candidates))
        flat_q = q_values.reshape((-1,))
        return candidate_actions[batch_indices, selected], selected, {
            "qsafe/rejected_fraction": jnp.mean(~safe_mask),
            "qsafe/fallback_fraction": jnp.mean(fallback),
            "qsafe/selected_value": jnp.mean(q_values[batch_indices, selected]),
            "qsafe/candidate_value_p50": jnp.quantile(flat_q, 0.50),
            "qsafe/candidate_value_p90": jnp.quantile(flat_q, 0.90),
            "qsafe/candidate_value_p99": jnp.quantile(flat_q, 0.99),
            "qsafe/candidate_value_min": jnp.min(flat_q),
            "qsafe/selected_log_probability": jnp.mean(
                log_probs[batch_indices, selected]
            ),
        }

    def select_safe_action(self, states, candidate_actions, candidate_log_probs, phase=None):
        phase = phase or self.phase
        if self.version == 2:
            states = self.normalize_observations(states)
        self.key, selection_key = jax.random.split(self.key)
        select = {
            "pretrain": self._select_pretrain_jit,
            "finetune": self._select_finetune_jit,
        }.get(phase)
        if select is None:
            raise ValueError(f"Unknown SQRL phase: {phase}")
        return select(
            self.state.params,
            states,
            candidate_actions,
            candidate_log_probs,
            selection_key,
        )

    def freeze(self):
        self.frozen = True
        self.observation_normalizer.freeze()

    def rollout_observations(self, observations, reset_mask=None, stream="task"):
        if self.version == 1:
            return np.asarray(observations, dtype=np.float32)
        values = np.asarray(observations, dtype=np.float32)
        history = self._rollout_histories.get(stream)
        if history is None or history.nr_envs != values.shape[0]:
            history = SafetyObservationHistory(
                values.shape[0], self.base_observation_shape[0], self.history_length
            )
            self._rollout_histories[stream] = history
        return history.append(values, reset_mask=reset_mask)

    def clear_rollout_history(self, stream=None):
        if stream is None:
            self._rollout_histories.clear()
        else:
            self._rollout_histories.pop(stream, None)

    def metadata(self):
        legacy = {
            "checkpoint_version": self.checkpoint_version,
            "observation_shape": list(self.observation_shape),
            "action_shape": list(self.action_shape),
            "observation_indices": self.observation_indices.tolist(),
            "nr_hidden_units": int(self.config.nr_hidden_units),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "max_trajectories": int(self.config.max_trajectories),
        }
        if self.version == 1:
            return legacy
        return {
            **legacy,
            "qsafe_version": self.version,
            "base_observation_shape": list(self.base_observation_shape),
            "history_length": self.history_length,
            "control_dt": self.control_dt,
            "history_duration": self.history_length * self.control_dt,
            "output_range": [0.0, 1.0],
            "safety_observation_contract": {
                "source": "raw_policy_observation",
                "frame_size": self.base_observation_shape[0],
                "layout": "oldest_to_newest_flattened",
                "reset_fill": "repeat_first_frame",
            },
            "safety_action_contract": {
                "source": "projected_executed_policy_action",
                "shape": list(self.action_shape),
                "range": [-1.0, 1.0],
                "semantics": "normalized_joint_position_target_after_rate_and_joint_limits",
            },
            "environment_contract": self.environment_contract,
            "normalizer": self.observation_normalizer.metadata(),
        }

    def state_dict(self, include_optimizer=True):
        train_state = serialization.to_state_dict(self.state)
        if not include_optimizer:
            train_state.pop("opt_state", None)
            train_state.pop("step", None)
        return {
            "metadata": self.metadata(),
            "config": self.config.to_dict(),
            "train_state": train_state,
            "safety_observation_normalizer_state": (
                self.observation_normalizer.state_dict()
            ),
            "safety_observation_normalizer_metadata": (
                self.observation_normalizer.metadata()
            ),
            "calibration_report": dict(self.calibration_report),
        }

    def _validate_metadata(self, metadata):
        expected = self.metadata()
        keys = [
            "checkpoint_version",
            "observation_shape",
            "action_shape",
            "observation_indices",
            "nr_hidden_units",
            "gamma",
            "epsilon",
            "max_trajectories",
        ]
        if self.version == 2:
            keys.extend(
                [
                    "qsafe_version",
                    "base_observation_shape",
                    "history_length",
                    "control_dt",
                    "output_range",
                    "environment_contract",
                ]
            )
        for key in keys:
            checkpoint_value = metadata[key]
            expected_value = expected[key]
            if isinstance(checkpoint_value, tuple):
                checkpoint_value = list(checkpoint_value)
            if isinstance(expected_value, tuple):
                expected_value = list(expected_value)
            if checkpoint_value != expected_value:
                raise ValueError(
                    f"Incompatible QSafe checkpoint metadata for {key}: "
                    f"expected {expected_value}, got {checkpoint_value}."
                )

    def load_state_dict(self, state, load_optimizer=True):
        self._validate_metadata(state["metadata"])
        if self.version == 2:
            normalizer_state = state.get("safety_observation_normalizer_state")
            normalizer_metadata = state.get(
                "safety_observation_normalizer_metadata"
            )
            if normalizer_state is None or normalizer_metadata is None:
                raise ValueError(
                    "QSafe v2 checkpoint is missing its independent normalizer."
                )
            self.observation_normalizer.load_state_dict(
                normalizer_state, normalizer_metadata
            )
            self.calibration_report = dict(state.get("calibration_report", {}))
        if load_optimizer:
            self.state = serialization.from_state_dict(self.state, state["train_state"])
        else:
            restored = state["train_state"]
            self.state = self.state.replace(
                params=serialization.from_state_dict(self.state.params, restored["params"]),
                target_params=serialization.from_state_dict(
                    self.state.target_params, restored["target_params"]
                ),
            )

    def save(self, file_path, include_optimizer=True):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        payload = self.state_dict(include_optimizer=include_optimizer)
        with open(file_path, "wb") as checkpoint_file:
            checkpoint_file.write(serialization.msgpack_serialize(payload))

    def load(self, file_path, load_optimizer=True):
        from rl_x.algorithms.sac_qsafe.flax.checkpoint import (
            load_torch_qsafe_artifact,
            looks_like_torch_checkpoint,
        )

        if looks_like_torch_checkpoint(file_path):
            if load_optimizer:
                raise ValueError(
                    "PyTorch optimizer state cannot be resumed in Flax; load it "
                    "for frozen fine-tuning with load_optimizer=False"
                )
            artifact = load_torch_qsafe_artifact(
                file_path,
                self.state.params,
                self.state.target_params,
                self.observation_indices,
            )
            self._validate_metadata(artifact["metadata"])
            self.state = self.state.replace(
                params=artifact["online_params"],
                target_params=artifact["target_params"],
            )
            if self.version == 2:
                self.observation_normalizer.load_state_dict(
                    artifact["normalizer_state"], artifact["normalizer_metadata"]
                )
                self.calibration_report = artifact["calibration_report"]
            return
        with open(file_path, "rb") as checkpoint_file:
            payload = serialization.msgpack_restore(checkpoint_file.read())
        self.load_state_dict(payload, load_optimizer=load_optimizer)
