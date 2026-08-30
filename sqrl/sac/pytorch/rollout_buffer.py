from collections import deque

import numpy as np


class RolloutBuffer:
    """Recent on-policy replay that atomically stores complete trajectories."""

    def __init__(self, capacity, nr_envs, os_shape, as_shape, rng, max_trajectories=10):
        self.capacity = max(1, int(capacity))
        self.nr_envs = int(nr_envs)
        self.field_shapes = (tuple(os_shape), tuple(os_shape), tuple(as_shape), (), (), ())
        self.rng = rng
        self.max_trajectories = max(1, int(max_trajectories))
        self.trajectories = deque()
        self.pending = [[] for _ in range(self.nr_envs)]
        self.size = 0

    @property
    def nr_trajectories(self):
        return len(self.trajectories)

    def add(self, states, next_states, actions, failures, terminations, truncations, remaining):
        completed = []
        for env_index in range(self.nr_envs):
            transition = tuple(
                np.array(field[env_index], copy=True)
                for field in (states, next_states, actions, failures, terminations, truncations)
            )
            self.pending[env_index].append(transition)
            if terminations[env_index] or truncations[env_index]:
                completed.append(self.pending[env_index])
                self.pending[env_index] = []

        completed = completed[:remaining]
        for trajectory in completed:
            self.add_trajectory(trajectory)
        if len(completed) == remaining:
            self.pending = [[] for _ in range(self.nr_envs)]
        return len(completed)

    def add_trajectory(self, trajectory):
        if not trajectory:
            raise ValueError("Cannot add an empty trajectory to D_safe")

        trajectory = tuple(
            np.asarray(field, dtype=np.float32).reshape((-1,) + shape)
            for field, shape in zip(zip(*trajectory), self.field_shapes)
        )
        failures, terminations, truncations = trajectory[3:]
        dones = terminations.astype(bool) | truncations.astype(bool)

        if not np.all((failures == 0) | (failures == 1)):
            raise ValueError("D_safe failures must be binary")
        if np.any(failures.astype(bool) & ~terminations.astype(bool)):
            raise ValueError("Every failure in D_safe must terminate its episode")
        if not dones[-1] or np.any(dones[:-1]):
            raise ValueError("D_safe only accepts one complete trajectory")

        trajectory_length = len(trajectory[0])
        while self.trajectories and (
            len(self.trajectories) >= self.max_trajectories or self.size + trajectory_length > self.capacity
        ):
            evicted = self.trajectories.popleft()
            self.size -= len(evicted[0])

        self.trajectories.append(trajectory)
        self.size += trajectory_length

    def sample(self, nr_samples):
        if self.size == 0:
            raise ValueError("Cannot sample an empty safety replay buffer")

        fields = tuple(np.concatenate(field) for field in zip(*self.trajectories))
        indices = self.rng.integers(self.size, size=nr_samples)
        return tuple(field[indices] for field in fields)
