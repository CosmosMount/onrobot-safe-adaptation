from collections import deque

import numpy as np


class SafetyReplayBuffer:
    """Transition replay populated only from completed safe-policy rollouts.

    SQRL's :math:`D_safe` stores recent on-policy trajectories. Capacity is
    enforced by evicting whole oldest trajectories, never by retaining partial
    fragments of an old episode.
    """

    def __init__(
        self,
        capacity,
        nr_envs,
        observation_shape,
        action_shape,
        rng,
        max_trajectories=None,
    ):
        self.observation_shape = tuple(observation_shape)
        self.action_shape = tuple(action_shape)
        self.capacity = max(1, int(capacity))
        self.nr_envs = nr_envs
        self.rng = rng
        self.max_trajectories = (
            None if max_trajectories is None else max(1, int(max_trajectories))
        )
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
        while (
            self.trajectories
            and self.max_trajectories is not None
            and len(self.trajectories) >= self.max_trajectories
        ):
            evicted = self.trajectories.popleft()
            self.size -= evicted[0].shape[0]
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


class TransitionSafetyReplayBuffer:
    """SORL ``D union D_safe`` replay.

    ``D`` is the ordinary transition ring.  In Algorithm 1 of SORL, ``D_safe``
    receives only the transition whose safety signal is one.  Pre-failure risk
    is propagated by the safety Bellman target from ``D``; duplicating the whole
    failed trajectory here would over-label ordinary, still-safe transitions.
    """

    def __init__(
        self,
        capacity,
        nr_envs,
        observation_shape,
        action_shape,
        rng,
        unsafe_fraction=0.5,
    ):
        self.observation_shape = tuple(observation_shape)
        self.action_shape = tuple(action_shape)
        self.capacity = max(1, int(capacity) // int(nr_envs))
        self.nr_envs = int(nr_envs)
        self.rng = rng
        self.unsafe_fraction = float(unsafe_fraction)
        if not 0.0 <= self.unsafe_fraction <= 1.0:
            raise ValueError("SORL unsafe replay fraction must be in [0, 1]")
        shape = (self.capacity, self.nr_envs)
        self.states = np.zeros(shape + self.observation_shape, dtype=np.float32)
        self.next_states = np.zeros(
            shape + self.observation_shape, dtype=np.float32
        )
        self.actions = np.zeros(shape + self.action_shape, dtype=np.float32)
        self.failures = np.zeros(shape, dtype=np.float32)
        self.terminations = np.zeros(shape, dtype=np.float32)
        self.truncations = np.zeros(shape, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.unsafe_transitions = deque()
        self.unsafe_size = 0


    @property
    def nr_transitions(self):
        return self.size * self.nr_envs


    @property
    def nr_trajectories(self):
        # Compatibility metric retained for existing log consumers.  Each
        # record is now exactly one unsafe transition, not a trajectory.
        return len(self.unsafe_transitions)


    @property
    def nr_unsafe_transitions(self):
        return self.unsafe_size


    def add(
        self,
        states,
        next_states,
        actions,
        failures,
        terminations,
        truncations,
    ):
        failures = np.asarray(failures, dtype=np.float32).reshape(self.nr_envs)
        if not np.all((failures == 0.0) | (failures == 1.0)):
            raise ValueError("SORL replay failures must be binary")
        terminations = np.asarray(terminations, dtype=np.float32).reshape(
            self.nr_envs
        )
        if np.any(failures.astype(bool) & ~terminations.astype(bool)):
            raise ValueError("SORL failures must terminate the transition")
        self.states[self.pos] = np.asarray(states, dtype=np.float32).reshape(
            (self.nr_envs,) + self.observation_shape
        )
        self.next_states[self.pos] = np.asarray(
            next_states, dtype=np.float32
        ).reshape((self.nr_envs,) + self.observation_shape)
        self.actions[self.pos] = np.asarray(actions, dtype=np.float32).reshape(
            (self.nr_envs,) + self.action_shape
        )
        self.failures[self.pos] = failures
        self.terminations[self.pos] = terminations
        self.truncations[self.pos] = np.asarray(
            truncations, dtype=np.float32
        ).reshape(self.nr_envs)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def add_unsafe_transition(self, transition):
        if len(transition) != 6:
            raise ValueError("SORL D_safe transition must contain six fields")
        state, next_state, action, failure, termination, truncation = transition
        failure = float(failure)
        termination = float(termination)
        truncation = float(truncation)
        if failure != 1.0:
            raise ValueError("SORL D_safe requires an unsafe transition")
        if termination != 1.0:
            raise ValueError("SORL unsafe transitions must terminate")
        normalized = (
            np.asarray(state, dtype=np.float32).reshape(self.observation_shape),
            np.asarray(next_state, dtype=np.float32).reshape(self.observation_shape),
            np.asarray(action, dtype=np.float32).reshape(self.action_shape),
            np.float32(failure),
            np.float32(termination),
            np.float32(truncation),
        )
        while len(self.unsafe_transitions) >= self.capacity:
            self.unsafe_transitions.popleft()
        self.unsafe_transitions.append(normalized)
        self.unsafe_size = len(self.unsafe_transitions)


    def _sample_unsafe(self, nr_samples):
        indices = self.rng.integers(self.unsafe_size, size=nr_samples)
        transitions = tuple(self.unsafe_transitions)
        return tuple(
            np.stack(
                [
                    transitions[index][field_index]
                    for index in indices
                ]
            )
            for field_index in range(6)
        )


    def sample(self, nr_samples):
        if self.size == 0:
            raise ValueError("Cannot sample an empty SORL safety replay")
        nr_samples = int(nr_samples)
        nr_unsafe = (
            min(nr_samples, int(round(nr_samples * self.unsafe_fraction)))
            if self.unsafe_size
            else 0
        )
        general_steps = self.rng.integers(
            self.size, size=nr_samples - nr_unsafe
        )
        general_envs = self.rng.integers(
            self.nr_envs, size=nr_samples - nr_unsafe
        )
        general = (
            self.states[general_steps, general_envs],
            self.next_states[general_steps, general_envs],
            self.actions[general_steps, general_envs],
            self.failures[general_steps, general_envs],
            self.terminations[general_steps, general_envs],
            self.truncations[general_steps, general_envs],
        )
        if not nr_unsafe:
            return general
        unsafe = self._sample_unsafe(nr_unsafe)
        permutation = self.rng.permutation(nr_samples)
        return tuple(
            np.concatenate((general_field, unsafe_field), axis=0)[permutation]
            for general_field, unsafe_field in zip(general, unsafe)
        )
