import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import optax

from rl_x.algorithms.qsafe.replay_buffer import SafetyReplayBuffer
from rl_x.algorithms.qsafe.common import VectorTrajectoryAccumulator, safety_bellman_target
from rl_x.algorithms.qsafe.flax.safety_critic import SafetyQNetwork
from rl_x.algorithms.qsafe.flax.train_state import QSafeTrainState


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
        self.rng = rng
        self.key = key
        self.epsilon = float(self.config.epsilon)
        self.gamma = float(self.config.gamma)
        self.tau = float(self.config.tau)
        self.batch_size = int(self.config.batch_size)
        self.candidate_actions = int(self.config.candidate_actions)
        if self.candidate_actions < 1:
            raise ValueError("algorithm.qsafe.candidate_actions must be at least 1.")
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
        self.key, online_key = jax.random.split(self.key)
        dummy_observation = jnp.zeros((1,) + self.observation_shape, dtype=jnp.float32)
        dummy_action = jnp.zeros((1,) + self.action_shape, dtype=jnp.float32)
        params = self.network.init(online_key, dummy_observation, dummy_action)
        # JAX arrays are immutable, so sharing this initial tree value is safe;
        # later TrainState replacements keep online and target trees separate.
        target_params = jax.tree.map(lambda value: value.copy(), params)
        qsafe_optimizer = (
            optax.adam(float(self.config.learning_rate))
            if self.phase == "pretrain"
            else optax.set_to_zero()
        )
        self.state = QSafeTrainState.create(
            apply_fn=self.network.apply,
            params=params,
            target_params=target_params,
            tx=qsafe_optimizer,
        )
        self.replay_buffer = SafetyReplayBuffer(
            int(self.config.buffer_size),
            config.environment.nr_envs,
            self.observation_shape,
            self.action_shape,
            rng,
        )
        self.trajectory_accumulator = VectorTrajectoryAccumulator(
            config.environment.nr_envs
        )
        self.frozen = self.phase == "finetune"
        if self.phase == "finetune" and not defer_checkpoint_load:
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
                self.replay_buffer.add_trajectory(trajectory)

    def add_trajectory(self, trajectory):
        if not self.frozen:
            self.replay_buffer.add_trajectory(trajectory)

    def ready_to_update(self, global_step=None):
        return not self.frozen and self.replay_buffer.nr_transitions > 0

    def update(self, policy_sampler):
        if self.frozen:
            raise RuntimeError("Frozen QSafe cannot be updated during fine-tuning.")
        states, next_states, actions, failures, terminations, truncations = [
            jnp.asarray(value, dtype=jnp.float32)
            for value in self.replay_buffer.sample(self.batch_size)
        ]
        self.key, action_key = jax.random.split(self.key)
        next_actions = policy_sampler(next_states, action_key)

        def loss_fn(params):
            predicted = self.network.apply(params, states, actions)
            next_q = self.network.apply(
                jax.lax.stop_gradient(self.state.target_params), next_states, next_actions
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
        )(self.state.params)
        self.state = self.state.apply_gradients(grads=gradients)
        self.state = self.state.replace(
            target_params=optax.incremental_update(
                self.state.params, self.state.target_params, self.tau
            )
        )
        return {
            "loss/qsafe_loss": float(loss),
            "gradients/qsafe_grad_norm": float(optax.tree.norm(gradients)),
            "qsafe/value": float(jnp.mean(predicted)),
            "qsafe/target": float(jnp.mean(target)),
        }

    def values(self, states, actions):
        return self.network.apply(self.state.params, states, actions)

    def select_safe_action(self, states, candidate_actions, candidate_log_probs, phase=None):
        phase = phase or self.phase
        nr_envs, nr_candidates = candidate_actions.shape[:2]
        repeated_states = jnp.repeat(states[:, None, :], nr_candidates, axis=1)
        q_values = self.network.apply(
            self.state.params,
            repeated_states.reshape((nr_envs * nr_candidates, -1)),
            candidate_actions.reshape((nr_envs * nr_candidates, -1)),
        ).reshape((nr_envs, nr_candidates))
        safe_mask = q_values < self.epsilon
        fallback = ~jnp.any(safe_mask, axis=1)

        if phase == "pretrain":
            scores = jnp.where(safe_mask, q_values, -jnp.inf)
            selected = jnp.argmax(scores, axis=1)
        elif phase == "finetune":
            # Practical SQRL: importance-sample accepted candidates according
            # to their log probability under the original policy distribution.
            logits = candidate_log_probs.reshape((nr_envs, nr_candidates))
            logits = jnp.where(safe_mask, logits, -jnp.inf)
            logits = jnp.where(fallback[:, None], jnp.zeros_like(logits), logits)
            self.key, selection_key = jax.random.split(self.key)
            keys = jax.random.split(selection_key, nr_envs)
            selected = jax.vmap(jax.random.categorical)(keys, logits)
        else:
            raise ValueError(f"Unknown SQRL phase: {phase}")

        lowest_risk = jnp.argmin(q_values, axis=1)
        selected = jnp.where(fallback, lowest_risk, selected)
        batch_indices = jnp.arange(nr_envs)
        log_probs = candidate_log_probs.reshape((nr_envs, nr_candidates))
        return candidate_actions[batch_indices, selected], selected, {
            "qsafe/rejected_fraction": float(jnp.mean(~safe_mask)),
            "qsafe/fallback_fraction": float(jnp.mean(fallback)),
            "qsafe/selected_value": float(jnp.mean(q_values[batch_indices, selected])),
            "qsafe/selected_log_probability": float(
                jnp.mean(log_probs[batch_indices, selected])
            ),
        }

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
        }

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
        ):
            if metadata[key] != expected[key]:
                raise ValueError(
                    f"Incompatible QSafe checkpoint metadata for {key}: "
                    f"expected {expected[key]}, got {metadata[key]}."
                )

    def load_state_dict(self, state, load_optimizer=True):
        self._validate_metadata(state["metadata"])
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
        with open(file_path, "rb") as checkpoint_file:
            payload = serialization.msgpack_restore(checkpoint_file.read())
        self.load_state_dict(payload, load_optimizer=load_optimizer)
