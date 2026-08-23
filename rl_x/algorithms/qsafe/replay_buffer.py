from collections import deque

import numpy as np


class SafetyReplayBuffer:
    """Transition replay populated only from completed safe-policy rollouts.

    SQRL's :math:`D_safe` stores recent on-policy trajectories. Capacity is
    enforced by evicting whole oldest trajectories, never by retaining partial
    fragments of an old episode.
    """

    def __init__(self, capacity, nr_envs, observation_shape, action_shape, rng):
        self.observation_shape = tuple(observation_shape)
        self.action_shape = tuple(action_shape)
        self.capacity = max(1, int(capacity))
        self.nr_envs = nr_envs
        self.rng = rng
        self.trajectories = deque()
        self.size = 0

    @property
    def nr_transitions(self):
        return self.size

    @property
    def nr_trajectories(self):
        return len(self.trajectories)

    def _normalize(self, states, next_states, actions, failures, terminations, truncations):
        states = np.asarray(states, dtype=np.float32).reshape((-1,) + self.observation_shape)
        next_states = np.asarray(next_states, dtype=np.float32).reshape(
            (-1,) + self.observation_shape
        )
        actions = np.asarray(actions, dtype=np.float32).reshape((-1,) + self.action_shape)
        failures = np.asarray(failures, dtype=np.float32).reshape(-1)
        terminations = np.asarray(terminations, dtype=np.float32).reshape(-1)
        truncations = np.asarray(truncations, dtype=np.float32).reshape(-1)
        nr_items = states.shape[0]
        fields = (states, next_states, actions, failures, terminations, truncations)
        if nr_items == 0:
            raise ValueError("Cannot add an empty trajectory to D_safe.")
        if not all(value.shape[0] == nr_items for value in fields):
            raise ValueError("All D_safe transition fields must have the same length.")
        return fields

    def _append_trajectory(self, trajectory):
        trajectory_length = trajectory[0].shape[0]
        while self.trajectories and self.size + trajectory_length > self.capacity:
            evicted = self.trajectories.popleft()
            self.size -= evicted[0].shape[0]
        # A single long episode is retained intact even if it exceeds the soft
        # transition capacity; splitting it would violate D_safe semantics.
        self.trajectories.append(trajectory)
        self.size += trajectory_length

    def add_trajectory(self, trajectory):
        """Atomically commit one completed rollout to D_safe."""
        if not trajectory:
            raise ValueError("Cannot add an empty trajectory to D_safe.")
        fields = list(zip(*trajectory))
        normalized = self._normalize(*fields)
        failures = normalized[3]
        terminations = normalized[4].astype(bool)
        truncations = normalized[5].astype(bool)
        if not np.all((failures == 0) | (failures == 1)):
            raise ValueError("D_safe trajectory failures must be binary.")
        if np.any(failures.astype(bool) & ~terminations):
            raise ValueError(
                "D_safe trajectory violates SQRL: failure must terminate."
            )
        dones = terminations | truncations
        if not dones[-1] or np.any(dones[:-1]):
            raise ValueError(
                "D_safe accepts exactly one complete trajectory whose only "
                "terminal transition is its final item."
            )
        self._append_trajectory(normalized)

    def sample(self, nr_samples):
        if self.size == 0:
            raise ValueError("Cannot sample an empty safety replay buffer.")
        flat_indices = self.rng.integers(self.size, size=nr_samples)
        lengths = np.fromiter(
            (trajectory[0].shape[0] for trajectory in self.trajectories),
            dtype=np.int64,
        )
        cumulative_lengths = np.cumsum(lengths)
        trajectory_indices = np.searchsorted(
            cumulative_lengths, flat_indices, side="right"
        )
        previous_cumulative = np.concatenate(([0], cumulative_lengths[:-1]))
        transition_indices = flat_indices - previous_cumulative[trajectory_indices]
        trajectories = tuple(self.trajectories)
        sampled_fields = []
        for field_index in range(6):
            sampled_fields.append(
                np.stack(
                    [
                        trajectories[trajectory_index][field_index][transition_index]
                        for trajectory_index, transition_index in zip(
                            trajectory_indices, transition_indices
                        )
                    ]
                )
            )
        return tuple(sampled_fields)
