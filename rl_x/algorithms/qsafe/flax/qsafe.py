import os
import logging
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import optax

from rl_x.algorithms.qsafe.replay_buffer import (
    SafetyReplayBuffer,
    TransitionSafetyReplayBuffer,
)
from rl_x.algorithms.qsafe.common import VectorTrajectoryAccumulator, safety_bellman_target
from rl_x.algorithms.qsafe.flax.safety_critic import SafetyQNetwork
from rl_x.algorithms.qsafe.flax.train_state import QSafeTrainState
from rl_x.algorithms.qsafe.flax.projection import resolve_action_projectors


rlx_logger = logging.getLogger("rl_x")


class QSafe:
    """Flax implementation of the task-critic-independent SQRL layer."""

    CHECKPOINT_VERSION = 1

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
        self.safety_objective = str(config.algorithm.safety_objective)
        self.sorl_enabled = (
            self.phase == "finetune" and self.safety_objective == "sorl"
        )
        self.rng = rng
        self.key = key
        self.epsilon = float(self.config.epsilon)
        self.gamma = float(self.config.gamma)
        self.tau = float(self.config.tau)
        self.batch_size = int(self.config.batch_size)
        self.candidate_actions = int(self.config.candidate_actions)
        if self.candidate_actions < 1:
            raise ValueError("algorithm.qsafe.candidate_actions must be at least 1.")
        self.finetune_selector = str(
            getattr(self.config, "finetune_selector", "importance")
        )
        if self.finetune_selector not in ("importance", "first_safe"):
            raise ValueError(
                "algorithm.qsafe.finetune_selector must be 'importance' or "
                "'first_safe'."
            )
        self.observation_shape = tuple(env.single_observation_space.shape)
        self.action_shape = tuple(env.single_action_space.shape)
        self.observation_indices = np.asarray(
            getattr(
                env,
                "safety_critic_observation_indices",
                np.arange(self.observation_shape[0]),
            ),
            dtype=np.int32,
        )
        self.network = SafetyQNetwork(
            self.observation_indices.tolist(), int(self.config.nr_hidden_units)
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
        self.key, online_key, second_key = jax.random.split(self.key, 3)
        dummy_observation = jnp.zeros((1,) + self.observation_shape, dtype=jnp.float32)
        dummy_action = jnp.zeros((1,) + self.action_shape, dtype=jnp.float32)
        params = self.network.init(online_key, dummy_observation, dummy_action)
        # JAX arrays are immutable, so sharing this initial tree value is safe;
        # later TrainState replacements keep online and target trees separate.
        target_params = jax.tree.map(lambda value: value.copy(), params)
        qsafe_optimizer = (
            optax.adam(float(self.config.learning_rate))
            if self.phase == "pretrain" or self.sorl_enabled
            else optax.set_to_zero()
        )
        self.state = QSafeTrainState.create(
            apply_fn=self.network.apply,
            params=params,
            target_params=target_params,
            tx=qsafe_optimizer,
        )
        self.second_state = None
        if self.sorl_enabled:
            second_params = self.network.init(
                second_key, dummy_observation, dummy_action
            )
            self.second_state = QSafeTrainState.create(
                apply_fn=self.network.apply,
                params=second_params,
                target_params=jax.tree.map(
                    lambda value: value.copy(), second_params
                ),
                tx=qsafe_optimizer,
            )
        self._update_jit = jax.jit(self._update_state)
        self._update_twin_jit = jax.jit(self._update_twin_states)
        self._values_jit = jax.jit(self.network.apply)
        self._conservative_values_jit = jax.jit(
            lambda first_params, second_params, states, actions: jnp.clip(
                jnp.maximum(
                    self.network.apply(first_params, states, actions),
                    self.network.apply(second_params, states, actions),
                ),
                0.0,
                1.0,
            )
        )
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
                first_safe=self.finetune_selector == "first_safe",
            )
        )
        if self.sorl_enabled:
            self.replay_buffer = TransitionSafetyReplayBuffer(
                int(self.config.buffer_size),
                config.environment.nr_envs,
                self.observation_shape,
                self.action_shape,
                rng,
                unsafe_fraction=float(
                    config.algorithm.sorl.unsafe_replay_fraction
                ),
            )
        else:
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
        self.frozen = self.phase == "finetune" and not self.sorl_enabled
        if self.phase == "finetune" and not defer_checkpoint_load:
            checkpoint_path = str(self.config.checkpoint_path)
            if not checkpoint_path:
                raise ValueError("algorithm.qsafe.checkpoint_path is required for finetune.")
            self.load(checkpoint_path, load_optimizer=False)
        if self.phase == "finetune" and not self.sorl_enabled:
            self.freeze()

    def add_transition(self, states, actions, next_states, failures, terminations, truncations):
        if self.sorl_enabled:
            self.replay_buffer.add(
                states,
                next_states,
                actions,
                failures,
                terminations,
                truncations,
            )
            failures = np.asarray(failures).reshape(-1)
            terminations = np.asarray(terminations).reshape(-1)
            truncations = np.asarray(truncations).reshape(-1)
            for env_index in np.flatnonzero(failures > 0.0):
                self.replay_buffer.add_unsafe_transition(
                    (
                        np.asarray(states)[env_index],
                        np.asarray(next_states)[env_index],
                        np.asarray(actions)[env_index],
                        failures[env_index],
                        terminations[env_index],
                        truncations[env_index],
                    )
                )
        elif not self.frozen:
            completed = self.trajectory_accumulator.add_step(
                states,
                next_states,
                actions,
                failures,
                terminations,
                truncations,
            )
            for trajectory in completed:
                self.replay_buffer.add_trajectory(trajectory)

    def add_trajectory(self, trajectory):
        if not self.frozen:
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


    def _update_twin_states(
        self,
        first_state,
        second_state,
        states,
        next_states,
        actions,
        failures,
        terminations,
        truncations,
        next_actions,
    ):
        first_next = self.network.apply(
            jax.lax.stop_gradient(first_state.target_params),
            next_states,
            next_actions,
        )
        second_next = self.network.apply(
            jax.lax.stop_gradient(second_state.target_params),
            next_states,
            next_actions,
        )
        next_risk = jnp.clip(jnp.maximum(first_next, second_next), 0.0, 1.0)
        target = jax.lax.stop_gradient(
            safety_bellman_target(
                failures[:, None],
                terminations[:, None],
                truncations[:, None],
                next_risk,
                self.gamma,
            )
        )

        def loss_fn(params):
            predicted = self.network.apply(params, states, actions)
            return jnp.mean((predicted - target) ** 2), predicted

        (first_loss, first_predicted), first_gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(first_state.params)
        (second_loss, second_predicted), second_gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(second_state.params)
        first_state = first_state.apply_gradients(grads=first_gradients)
        second_state = second_state.apply_gradients(grads=second_gradients)
        first_state = first_state.replace(
            target_params=optax.incremental_update(
                first_state.params, first_state.target_params, self.tau
            )
        )
        second_state = second_state.replace(
            target_params=optax.incremental_update(
                second_state.params, second_state.target_params, self.tau
            )
        )
        return first_state, second_state, {
            "loss/qsafe_loss": 0.5 * (first_loss + second_loss),
            "loss/sorl_safety_first": first_loss,
            "loss/sorl_safety_second": second_loss,
            "gradients/qsafe_grad_norm": 0.5
            * (
                optax.tree.norm(first_gradients)
                + optax.tree.norm(second_gradients)
            ),
            "qsafe/value": jnp.mean(
                jnp.maximum(first_predicted, second_predicted)
            ),
            "qsafe/target": jnp.mean(target),
            "sorl/safety_disagreement": jnp.mean(
                jnp.abs(first_predicted - second_predicted)
            ),
        }

    def update(self, policy_sampler, state_transform=None):
        if self.frozen:
            raise RuntimeError("Frozen QSafe cannot be updated during fine-tuning.")
        raw_states, raw_next_states, actions, failures, terminations, truncations = [
            jnp.asarray(value, dtype=jnp.float32)
            for value in self.replay_buffer.sample(self.batch_size)
        ]
        states = raw_states
        next_states = raw_next_states
        if state_transform is not None:
            states = state_transform(states)
            next_states = state_transform(next_states)
        self.key, action_key = jax.random.split(self.key)
        next_actions = policy_sampler(next_states, action_key)
        if self._projector_is_jax:
            next_actions = self._jax_project_actions(raw_next_states, next_actions)
        elif self._host_project_actions is not None:
            next_actions = jnp.asarray(
                self._host_project_actions(
                    jax.device_get(raw_next_states), jax.device_get(next_actions)
                ),
                dtype=jnp.float32,
            )
        if self.sorl_enabled:
            self.state, self.second_state, metrics = self._update_twin_jit(
                self.state,
                self.second_state,
                states,
                next_states,
                actions,
                failures,
                terminations,
                truncations,
                next_actions,
            )
        else:
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


    def warm_up_sorl_update(self, policy_sampler, state_transform=None):
        if not self.sorl_enabled:
            return
        raw_states = jnp.zeros(
            (self.batch_size,) + self.observation_shape, dtype=jnp.float32
        )
        states = raw_states
        if state_transform is not None:
            states = state_transform(states)
        self.key, action_key = jax.random.split(self.key)
        next_actions = policy_sampler(states, action_key)
        if self._projector_is_jax:
            next_actions = self._jax_project_actions(raw_states, next_actions)
        failures = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        result = self._update_twin_jit(
            self.state,
            self.second_state,
            states,
            states,
            jnp.zeros(
                (self.batch_size,) + self.action_shape, dtype=jnp.float32
            ),
            failures,
            failures,
            failures,
            next_actions,
        )
        jax.block_until_ready(result[2])

    def values(self, states, actions):
        return self._values_jit(self.state.params, states, actions)


    def conservative_values(self, states, actions):
        """Return a bounded online/target envelope for SORL reward shaping."""

        second_params = (
            self.second_state.params
            if self.second_state is not None
            else self.state.target_params
        )
        return self._conservative_values_jit(
            self.state.params, second_params, states, actions
        )

    def _select_kernel(
        self,
        params,
        states,
        candidate_actions,
        candidate_log_probs,
        selection_key,
        *,
        pretrain,
        first_safe=False,
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
        elif first_safe:
            selected = jnp.argmax(safe_mask, axis=1)
        else:
            logits = candidate_log_probs.reshape((nr_envs, nr_candidates))
            logits = jnp.where(safe_mask, logits, -jnp.inf)
            logits = jnp.where(fallback[:, None], jnp.zeros_like(logits), logits)
            selected = jax.random.categorical(selection_key, logits, axis=-1)

        lowest_risk = jnp.argmin(q_values, axis=1)
        selected = jnp.where(fallback, lowest_risk, selected)
        batch_indices = jnp.arange(nr_envs)
        log_probs = candidate_log_probs.reshape((nr_envs, nr_candidates))
        return candidate_actions[batch_indices, selected], selected, {
            "qsafe/rejected_fraction": jnp.mean(~safe_mask),
            "qsafe/fallback_fraction": jnp.mean(fallback),
            "qsafe/selected_value": jnp.mean(q_values[batch_indices, selected]),
            "qsafe/selected_log_probability": jnp.mean(
                log_probs[batch_indices, selected]
            ),
        }

    def select_safe_action(self, states, candidate_actions, candidate_log_probs, phase=None):
        phase = phase or self.phase
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

    def metadata(self):
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "observation_shape": list(self.observation_shape),
            "action_shape": list(self.action_shape),
            "observation_indices": self.observation_indices.tolist(),
            "nr_hidden_units": int(self.config.nr_hidden_units),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "max_trajectories": int(self.config.max_trajectories),
            "safety_objective": self.safety_objective,
        }

    def state_dict(self, include_optimizer=True):
        train_state = serialization.to_state_dict(self.state)
        if not include_optimizer:
            train_state.pop("opt_state", None)
            train_state.pop("step", None)
        state = {
            "metadata": self.metadata(),
            "config": self.config.to_dict(),
            "train_state": train_state,
        }
        if self.second_state is not None:
            second_train_state = serialization.to_state_dict(self.second_state)
            if not include_optimizer:
                second_train_state.pop("opt_state", None)
                second_train_state.pop("step", None)
            state["second_train_state"] = second_train_state
        return state

    def _validate_metadata(self, metadata):
        expected = self.metadata()
        for key in (
            "checkpoint_version",
            "observation_shape",
            "action_shape",
            "observation_indices",
            "nr_hidden_units",
            "gamma",
            "epsilon",
            "max_trajectories",
        ):
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
        if load_optimizer:
            self.state = serialization.from_state_dict(self.state, state["train_state"])
            if self.second_state is not None and "second_train_state" in state:
                self.second_state = serialization.from_state_dict(
                    self.second_state, state["second_train_state"]
                )
        else:
            restored = state["train_state"]
            self.state = self.state.replace(
                params=serialization.from_state_dict(self.state.params, restored["params"]),
                target_params=serialization.from_state_dict(
                    self.state.target_params, restored["target_params"]
                ),
            )
            if self.second_state is not None:
                restored_second = state.get("second_train_state")
                if restored_second is None:
                    self.second_state = self.second_state.replace(
                        params=self.state.target_params,
                        target_params=self.state.target_params,
                    )
                else:
                    self.second_state = self.second_state.replace(
                        params=serialization.from_state_dict(
                            self.second_state.params,
                            restored_second["params"],
                        ),
                        target_params=serialization.from_state_dict(
                            self.second_state.target_params,
                            restored_second["target_params"],
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
            if self.second_state is not None:
                self.second_state = self.second_state.replace(
                    params=artifact["target_params"],
                    target_params=artifact["target_params"],
                )
            return
        with open(file_path, "rb") as checkpoint_file:
            payload = serialization.msgpack_restore(checkpoint_file.read())
        self.load_state_dict(payload, load_optimizer=load_optimizer)
