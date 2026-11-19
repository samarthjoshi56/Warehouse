"""Bounded replay buffer for masked discrete-action transitions."""
from __future__ import annotations
import json
from pathlib import Path
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
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done, next_mask):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.next_masks[i] = next_mask
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size > self.size:
            raise ValueError("batch_size exceeds current replay size")
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        return {"states": self.states[idx], "actions": self.actions[idx], "rewards": self.rewards[idx],
                "next_states": self.next_states[idx], "dones": self.dones[idx], "next_masks": self.next_masks[idx]}

    def save(self, path: str | Path) -> None:
        """Persist occupied transitions and sampling RNG, avoiding empty capacity."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Until the buffer wraps, all occupied entries are in [0:size).
        # Once full, every slot is occupied and ring position is preserved.
        stored = self.size
        np.savez_compressed(
            path, capacity=np.asarray(self.capacity), position=np.asarray(self.position),
            size=np.asarray(self.size), stored=np.asarray(stored),
            states=self.states[:stored], next_states=self.next_states[:stored],
            actions=self.actions[:stored], rewards=self.rewards[:stored], dones=self.dones[:stored],
            next_masks=self.next_masks[:stored],
            rng_state=np.asarray(json.dumps(self.rng.bit_generator.state)))

    def load(self, path: str | Path) -> None:
        """Restore buffer data. Accept compact or legacy full-size files."""
        with np.load(path, allow_pickle=False) as data:
            if int(data["capacity"]) != self.capacity:
                raise ValueError("Replay capacity mismatch")
            size = int(data["size"])
            position = int(data["position"])
            if not (0 <= position < self.capacity and 0 <= size <= self.capacity):
                raise ValueError("Invalid replay metadata")
            stored = int(data["stored"]) if "stored" in data else size
            if stored != size:
                raise ValueError("Stored transition count mismatch")
            for name in ("states", "next_states", "actions", "rewards", "dones", "next_masks"):
                target = getattr(self, name)
                source = data[name]
                expected = target.shape if size == self.capacity else (size, *target.shape[1:])
                if source.shape not in (expected, target.shape):
                    raise ValueError(f"Replay shape mismatch for {name}: {source.shape} != {expected}")
                target.fill(0)
                target[:size] = source[:size]
            self.position = position
            self.size = size
            self.rng.bit_generator.state = json.loads(str(data["rng_state"]))

    def __len__(self):
        return self.size
