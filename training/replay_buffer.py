"""Bounded replay buffer for masked discrete-action transitions."""
from __future__ import annotations
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, action_dim), dtype=bool)
        self.position = 0; self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done, next_mask):
        i = self.position
        self.states[i] = state; self.actions[i] = action; self.rewards[i] = reward
        self.next_states[i] = next_state; self.dones[i] = float(done); self.next_masks[i] = next_mask
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size > self.size:
            raise ValueError("batch_size exceeds current replay size")
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        return {"states": self.states[idx], "actions": self.actions[idx], "rewards": self.rewards[idx],
                "next_states": self.next_states[idx], "dones": self.dones[idx], "next_masks": self.next_masks[idx]}

    def __len__(self):
        return self.size
