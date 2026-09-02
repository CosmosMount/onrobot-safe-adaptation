import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from rl_x.algorithms.qsafe.common import (
    SafetyObservationHistory,
    VectorTrajectoryAccumulator,
    safety_bellman_target,
    trajectory_with_observation_history,
)
from rl_x.algorithms.qsafe.replay_buffer import SafetyReplayBuffer
from rl_x.algorithms.qsafe.pytorch.safety_critic import SafetyQNetwork


class SafetyObservationNormalizer(nn.Module):
    """QSafe-owned running statistics; never shared with the actor."""

    def __init__(self, observation_size, enabled=True, epsilon=1e-8):
        super().__init__()
        self.observation_size = int(observation_size)
        self.enabled = bool(enabled)
        self.epsilon = float(epsilon)
        self.register_buffer("running_mean", torch.zeros(1, observation_size))
        self.register_buffer("running_var", torch.ones(1, observation_size))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.frozen = False

    @property
    def running_std(self):
        return torch.sqrt(torch.clamp(self.running_var, min=0.0))

    @torch.no_grad()
    def update(self, observations):
        if not self.enabled or self.frozen:
            return
        values = observations.detach().reshape(-1, self.observation_size).float()
        if not values.shape[0]:
            return
        batch_count = float(values.shape[0])
        batch_mean = values.mean(dim=0, keepdim=True)
        batch_var = values.var(dim=0, unbiased=False, keepdim=True)
        if self.count.item() == 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
            self.count.fill_(batch_count)
            return
        old_count = self.count.to(dtype=values.dtype)
        new_count = old_count + batch_count
        delta = batch_mean - self.running_mean
        mean = self.running_mean + delta * batch_count / new_count
        m2 = (
            self.running_var * old_count
            + batch_var * batch_count
            + delta.square() * old_count * batch_count / new_count
        )
        self.running_mean.copy_(mean)
        self.running_var.copy_(m2 / new_count)
        self.count.fill_(float(new_count.item()))

    def normalize(self, observations, update=False):
        if update:
            self.update(observations)
        values = observations.float()
        if not self.enabled:
            return values
        return (values - self.running_mean) / (self.running_std + self.epsilon)

    def freeze(self):
        self.frozen = True
        self.eval()

    def metadata(self):
        return {
            "observation_size": self.observation_size,
            "enabled": self.enabled,
            "epsilon": self.epsilon,
            "count": int(self.count.item()),
        }

    def validate_metadata(self, metadata):
        for key in ("observation_size", "enabled", "epsilon"):
            if metadata.get(key) != self.metadata()[key]:
                raise ValueError(
                    f"Incompatible safety normalizer {key}: expected "
                    f"{self.metadata()[key]}, got {metadata.get(key)}"
                )


class QSafe:
    """Reusable SQRL safety critic and action-projection layer."""

    def __init__(
        self,
        config,
        env,
        device,
        rng,
        phase=None,
        defer_checkpoint_load=False,
    ):
        self.config = config.algorithm.qsafe
        self.phase = phase or config.algorithm.phase
        self.device = device
        self.rng = rng
        self.version = int(getattr(self.config, "version", 1))
        if self.version not in (1, 2):
            raise ValueError("algorithm.qsafe.version must be 1 or 2.")
        configured_selection_mode = str(
            getattr(self.config, "selection_mode", "auto")
        )
        self.selection_mode = configured_selection_mode
        if self.selection_mode not in (
            "auto",
            "legacy_density_resample",
            "rejection_sampling",
        ):
            raise ValueError(
                "algorithm.qsafe.selection_mode must be 'auto', "
                "'legacy_density_resample', or 'rejection_sampling'."
            )

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
        if self.control_dt <= 0.0:
            raise ValueError("algorithm.qsafe.control_dt must be positive.")
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
                dtype=np.int64,
            )
        else:
            self.observation_indices = np.arange(
                self.observation_shape[0], dtype=np.int64
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
        ).to(device)
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
        hidden_units = int(self.config.nr_hidden_units)
        self.online = SafetyQNetwork(
            self.observation_shape,
            self.action_shape,
            self.observation_indices,
            hidden_units,
            self.output_activation,
        ).to(device)
        self.target = SafetyQNetwork(
            self.observation_shape,
            self.action_shape,
            self.observation_indices,
            hidden_units,
            self.output_activation,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = None
        if self.phase == "pretrain" and bool(config.algorithm.qsafe.enabled):
            self.optimizer = torch.optim.Adam(
                self.online.parameters(), lr=float(self.config.learning_rate)
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

    def resolved_selection_mode(self):
        configured = getattr(self, "selection_mode", "auto")
        if configured == "auto":
            return (
                "legacy_density_resample"
                if self.version == 1
                else "rejection_sampling"
            )
        return configured

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
                states = torch.as_tensor(
                    np.stack([item[0] for item in trajectory]),
                    dtype=torch.float32,
                    device=self.device,
                )
                next_states = torch.as_tensor(
                    np.stack([item[1] for item in trajectory]),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.observation_normalizer.update(
                    torch.cat([states, next_states], dim=0)
                )
            self.replay_buffer.add_trajectory(trajectory)

    def ready_to_update(self, global_step=None):
        # Algorithm 1 updates after every D_safe rollout block. Sampling with
        # replacement permits the configured batch size from the first block.
        return not self.frozen and self.replay_buffer.nr_transitions > 0

    def update(self, policy_sampler, state_transform=None, action_transform=None):
        if self.frozen or self.optimizer is None:
            raise RuntimeError("Frozen QSafe cannot be updated during fine-tuning.")
        arrays = self.replay_buffer.sample(self.batch_size)
        states, next_states, actions, failures, terminations, truncations = [
            torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for value in arrays
        ]
        raw_next_states = next_states
        if self.version == 2:
            states = self.observation_normalizer.normalize(states, update=False)
            next_states = self.observation_normalizer.normalize(
                next_states, update=False
            )
            policy_next_states = raw_next_states[
                ..., -self.base_observation_shape[0] :
            ]
            if state_transform is not None:
                policy_next_states = state_transform(policy_next_states)
        else:
            policy_next_states = next_states
            if state_transform is not None:
                states = state_transform(states)
                next_states = state_transform(next_states)
                policy_next_states = next_states
        with torch.no_grad():
            next_actions = policy_sampler(policy_next_states)
            if action_transform is not None:
                action_raw_states = (
                    raw_next_states[..., -self.base_observation_shape[0] :]
                    if self.version == 2
                    else raw_next_states
                )
                next_actions = action_transform(action_raw_states, next_actions)
            next_q = self.target(next_states, next_actions)
            failures = failures.reshape(-1, 1)
            target = safety_bellman_target(
                failures,
                terminations.reshape(-1, 1),
                truncations.reshape(-1, 1),
                next_q,
                self.gamma,
            )

        predicted = self.online(states, actions)
        loss = F.mse_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), float("inf"))
        self.optimizer.step()
        with torch.no_grad():
            for parameter, target_parameter in zip(
                self.online.parameters(), self.target.parameters()
            ):
                target_parameter.data.mul_(1.0 - self.tau).add_(
                    parameter.data, alpha=self.tau
                )
        return {
            "loss/qsafe_loss": loss.detach().item(),
            "gradients/qsafe_grad_norm": grad_norm.detach().item(),
            "qsafe/value": predicted.detach().mean().item(),
            "qsafe/target": target.detach().mean().item(),
        }

    def normalize_observations(self, states):
        if self.version == 2:
            return self.observation_normalizer.normalize(states, update=False)
        return states

    def values(self, states, actions, normalized=False):
        if self.version == 2 and not normalized:
            states = self.normalize_observations(states)
        return self.online(states, actions)

    def select_safe_action(self, states, candidate_actions, candidate_log_probs, phase=None):
        """Project sampled actions; candidate tensors are [env, candidate, ...]."""
        phase = phase or self.phase
        if self.version == 2:
            states = self.normalize_observations(states)
        nr_envs, nr_candidates = candidate_actions.shape[:2]
        repeated_states = states[:, None, :].expand(-1, nr_candidates, -1)
        with torch.no_grad():
            q_values = self.online(
                repeated_states.reshape(nr_envs * nr_candidates, -1),
                candidate_actions.reshape(nr_envs * nr_candidates, -1),
            ).reshape(nr_envs, nr_candidates)
        safe_mask = q_values < self.epsilon
        fallback = ~safe_mask.any(dim=1)

        if phase == "pretrain":
            boundary_scores = torch.where(
                safe_mask, q_values, torch.full_like(q_values, -torch.inf)
            )
            selected = boundary_scores.argmax(dim=1)
        elif (
            phase == "finetune"
            and self.resolved_selection_mode() == "legacy_density_resample"
        ):
            # Historical implementation retained for reproducing completed
            # experiments. The second policy-density weighting changes the
            # already sampled candidate distribution from pi toward pi^2.
            logits = candidate_log_probs.reshape(nr_envs, nr_candidates)
            masked_logits = torch.where(
                safe_mask, logits, torch.full_like(logits, -torch.inf)
            )
            masked_logits = torch.where(
                fallback[:, None], torch.zeros_like(masked_logits), masked_logits
            )
            selected = torch.distributions.Categorical(
                logits=masked_logits
            ).sample()
        elif phase == "finetune":
            # Candidates are IID samples from the task policy. The first
            # accepted sample is exact finite rejection sampling from Eq. 3.
            candidate_indices = torch.arange(
                nr_candidates, device=candidate_actions.device
            )[None, :].expand(nr_envs, -1)
            selected = torch.where(
                safe_mask,
                candidate_indices,
                torch.full_like(candidate_indices, nr_candidates),
            ).min(dim=1).values
        else:
            raise ValueError(f"Unknown SQRL phase: {phase}")

        lowest_risk = q_values.argmin(dim=1)
        selected = torch.where(fallback, lowest_risk, selected)
        batch_indices = torch.arange(nr_envs, device=candidate_actions.device)
        log_probs = candidate_log_probs.reshape(nr_envs, nr_candidates)
        flat_q = q_values.reshape(-1)
        selected_actions = candidate_actions[batch_indices, selected]
        candidate_zero_rejected = ~safe_mask[:, 0]
        action_delta = selected_actions - candidate_actions[:, 0]
        action_changed = torch.linalg.vector_norm(
            action_delta.reshape(nr_envs, -1), dim=-1
        ) > 1e-6
        return selected_actions, selected, {
            "qsafe/rejected_fraction": (~safe_mask).float().mean().item(),
            "qsafe/fallback_fraction": fallback.float().mean().item(),
            "qsafe/action_change_fraction": action_changed.float().mean().item(),
            "qsafe/action_change_l2": torch.linalg.vector_norm(
                action_delta.reshape(nr_envs, -1), dim=-1
            ).mean().item(),
            "qsafe/candidate0_rejected_fraction": (
                candidate_zero_rejected.float().mean().item()
            ),
            "qsafe/safety_intervention_fraction": (
                (action_changed & candidate_zero_rejected).float().mean().item()
            ),
            "qsafe/selected_value": q_values[batch_indices, selected].mean().item(),
            "qsafe/candidate_value_p50": torch.quantile(flat_q, 0.50).item(),
            "qsafe/candidate_value_p90": torch.quantile(flat_q, 0.90).item(),
            "qsafe/candidate_value_p99": torch.quantile(flat_q, 0.99).item(),
            "qsafe/candidate_value_min": flat_q.min().item(),
            "qsafe/selected_log_probability": log_probs[
                batch_indices, selected
            ].mean().item(),
        }

    def freeze(self):
        self.frozen = True
        self.online.eval()
        self.target.eval()
        for parameter in self.online.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = None
        self.observation_normalizer.freeze()

    def rollout_observations(self, observations, reset_mask=None, stream="task"):
        """Append one raw 46D frame and return the QSafe input for a stream."""

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
            "observation_shape": self.observation_shape,
            "action_shape": self.action_shape,
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
            "base_observation_shape": self.base_observation_shape,
            "history_length": self.history_length,
            "control_dt": self.control_dt,
            "history_duration": self.history_length * self.control_dt,
            "output_range": [0.0, 1.0] if self.version == 2 else [-1.0, 1.0],
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
        state = {
            "metadata": self.metadata(),
            "config": self.config.to_dict(),
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "safety_observation_normalizer_state_dict": (
                self.observation_normalizer.state_dict()
            ),
            "safety_observation_normalizer_metadata": (
                self.observation_normalizer.metadata()
            ),
            "calibration_report": dict(self.calibration_report),
        }
        if include_optimizer and self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        return state

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
            if isinstance(checkpoint_value, list):
                checkpoint_value = tuple(checkpoint_value)
            if isinstance(expected_value, list):
                expected_value = tuple(expected_value)
            if checkpoint_value != expected_value:
                raise ValueError(
                    f"Incompatible QSafe checkpoint metadata for {key}: "
                    f"expected {expected_value}, got {checkpoint_value}."
                )

    def load_state_dict(self, state, load_optimizer=True):
        self._validate_metadata(state["metadata"])
        self.online.load_state_dict(state["online_state_dict"])
        self.target.load_state_dict(state["target_state_dict"])
        if self.version == 2:
            normalizer_state = state.get(
                "safety_observation_normalizer_state_dict"
            )
            normalizer_metadata = state.get(
                "safety_observation_normalizer_metadata"
            )
            if normalizer_state is None or normalizer_metadata is None:
                raise ValueError(
                    "QSafe v2 checkpoint is missing its independent normalizer."
                )
            self.observation_normalizer.validate_metadata(normalizer_metadata)
            self.observation_normalizer.load_state_dict(normalizer_state)
            self.calibration_report = dict(state.get("calibration_report", {}))
        if load_optimizer and self.optimizer is not None and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

    def save(self, file_path, include_optimizer=True):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save(self.state_dict(include_optimizer=include_optimizer), file_path)

    def load(self, file_path, load_optimizer=True):
        state = torch.load(file_path, map_location=self.device, weights_only=False)
        self.load_state_dict(state, load_optimizer=load_optimizer)
