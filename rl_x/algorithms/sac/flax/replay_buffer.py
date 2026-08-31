import numpy as np


class ReplayBuffer():
    def __init__(
        self,
        capacity,
        nr_envs,
        os_shape,
        as_shape,
        rng,
        auxiliary_state_shape=None,
    ):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.capacity = capacity // nr_envs
        self.nr_envs = nr_envs
        self.rng = rng
        self.auxiliary_state_shape = (
            None if auxiliary_state_shape is None else tuple(auxiliary_state_shape)
        )
        self.states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.terminations = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.auxiliary_states = (
            None
            if self.auxiliary_state_shape is None
            else np.zeros(
                (self.capacity, nr_envs) + self.auxiliary_state_shape,
                dtype=np.float32,
            )
        )
        self.pos = 0
        self.size = 0
    

    def add(
        self,
        states,
        next_states,
        actions,
        rewards,
        terminations,
        auxiliary_states=None,
    ):
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.terminations[self.pos] = terminations
        if self.auxiliary_states is not None:
            if auxiliary_states is None:
                raise ValueError("Replay buffer requires auxiliary_states.")
            self.auxiliary_states[self.pos] = auxiliary_states
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    

    def sample(self, nr_samples):
        idx1 = self.rng.integers(self.size, size=nr_samples)
        idx2 = self.rng.integers(self.nr_envs, size=nr_samples)
        states = self.states[idx1, idx2].reshape((nr_samples,) + self.os_shape)
        next_states = self.next_states[idx1, idx2].reshape((nr_samples,) + self.os_shape)
        actions = self.actions[idx1, idx2].reshape((nr_samples,) + self.as_shape)
        rewards = self.rewards[idx1, idx2].reshape((nr_samples,))
        terminations = self.terminations[idx1, idx2].reshape((nr_samples,))
        result = (states, next_states, actions, rewards, terminations)
        if self.auxiliary_states is not None:
            auxiliary_states = self.auxiliary_states[idx1, idx2].reshape(
                (nr_samples,) + self.auxiliary_state_shape
            )
            result = (*result, auxiliary_states)
        return result
