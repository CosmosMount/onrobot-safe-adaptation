import os
import numpy as np
import torch
import torch.nn.functional as F

from rl_x.algorithms.qsafe.common import VectorTrajectoryAccumulator, safety_bellman_target
from rl_x.algorithms.qsafe.replay_buffer import SafetyReplayBuffer
from rl_x.algorithms.qsafe.pytorch.safety_critic import SafetyQNetwork


class QSafe:
    """Reusable SQRL safety critic and action-projection layer."""

    CHECKPOINT_VERSION = 1

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
            dtype=np.int64,
        )
        hidden_units = int(self.config.nr_hidden_units)
        self.online = SafetyQNetwork(
            self.observation_shape,
            self.action_shape,
            self.observation_indices,
            hidden_units,
        ).to(device)
        self.target = SafetyQNetwork(
            self.observation_shape,
            self.action_shape,
            self.observation_indices,
            hidden_units,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = None
        if self.phase == "pretrain":
            self.optimizer = torch.optim.Adam(
                self.online.parameters(), lr=float(self.config.learning_rate)
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
        if state_transform is not None:
            states = state_transform(states)
            next_states = state_transform(next_states)
        with torch.no_grad():
            next_actions = policy_sampler(next_states)
            if action_transform is not None:
                next_actions = action_transform(raw_next_states, next_actions)
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

    def values(self, states, actions):
        return self.online(states, actions)

    def select_safe_action(self, states, candidate_actions, candidate_log_probs, phase=None):
        """Project sampled actions; candidate tensors are [env, candidate, ...]."""
        phase = phase or self.phase
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
        elif phase == "finetune":
            # Practical SQRL: mask unsafe samples, then importance-sample the
            # remaining candidates according to their likelihood under pi.
            logits = candidate_log_probs.reshape(nr_envs, nr_candidates)
            masked_logits = torch.where(
                safe_mask, logits, torch.full_like(logits, -torch.inf)
            )
            masked_logits = torch.where(
                fallback[:, None], torch.zeros_like(masked_logits), masked_logits
            )
            selected = torch.distributions.Categorical(logits=masked_logits).sample()
        else:
            raise ValueError(f"Unknown SQRL phase: {phase}")

        lowest_risk = q_values.argmin(dim=1)
        selected = torch.where(fallback, lowest_risk, selected)
        batch_indices = torch.arange(nr_envs, device=candidate_actions.device)
        log_probs = candidate_log_probs.reshape(nr_envs, nr_candidates)
        return candidate_actions[batch_indices, selected], selected, {
            "qsafe/rejected_fraction": (~safe_mask).float().mean().item(),
            "qsafe/fallback_fraction": fallback.float().mean().item(),
            "qsafe/selected_value": q_values[batch_indices, selected].mean().item(),
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

    def metadata(self):
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "observation_shape": self.observation_shape,
            "action_shape": self.action_shape,
            "observation_indices": self.observation_indices.tolist(),
            "nr_hidden_units": int(self.config.nr_hidden_units),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }

    def state_dict(self, include_optimizer=True):
        state = {
            "metadata": self.metadata(),
            "config": self.config.to_dict(),
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
        }
        if include_optimizer and self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
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
        ):
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
        if load_optimizer and self.optimizer is not None and "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

    def save(self, file_path, include_optimizer=True):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save(self.state_dict(include_optimizer=include_optimizer), file_path)

    def load(self, file_path, load_optimizer=True):
        state = torch.load(file_path, map_location=self.device, weights_only=False)
        self.load_state_dict(state, load_optimizer=load_optimizer)
