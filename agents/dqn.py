"""Masked Deep Q-Network agent for the warehouse placement task."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

def flatten_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    fields = ("grid", "incoming_item", "temp_storage", "stack_summaries", "schedule_lookahead", "sim_time")
    return np.concatenate([np.asarray(observation[k], dtype=np.float32).ravel() for k in fields])

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, action_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.995
    batch_size: int = 32
    replay_capacity: int = 50_000
    learning_starts: int = 500
    train_frequency: int = 16
    target_update_interval: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    hidden_dim: int = 64
    gradient_clip: float = 10.0

class DQNAgent:
    def __init__(self, observation_dim: int, action_dim: int, config: DQNConfig | None = None,
                 device: str | None = None, seed: int = 0):
        self.config = config or DQNConfig()
        self.action_dim = int(action_dim)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(observation_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target = QNetwork(observation_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.steps = 0
        self.updates = 0

    def reset(self) -> None:
        """Reset episode-local state while keeping learned weights and exploration progress."""
        return None

    def epsilon(self) -> float:
        c = self.config
        fraction = min(1.0, self.steps / max(1, c.epsilon_decay_steps))
        return c.epsilon_start + fraction * (c.epsilon_end - c.epsilon_start)

    def select_action(self, observation: dict[str, np.ndarray], action_mask=None, explore: bool = True) -> int:
        mask = np.asarray(observation["action_mask"] if action_mask is None else action_mask, dtype=bool)
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError("No valid action is available")
        if explore and self.rng.random() < self.epsilon():
            action = int(self.rng.choice(valid))
        else:
            state = torch.as_tensor(flatten_observation(observation), dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.online(state).squeeze(0)
                q_values[torch.as_tensor(~mask, dtype=torch.bool, device=self.device)] = -torch.inf
                action = int(torch.argmax(q_values).item())
        self.steps += 1
        return action

    def update(self, batch: dict[str, np.ndarray]) -> float:
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)
        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_online = self.online(next_states).masked_fill(~next_masks, -torch.inf)
            no_valid = ~next_masks.any(dim=1)
            next_actions = next_online.argmax(dim=1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q[no_valid] = 0.0
            target = rewards + self.config.gamma * (1.0 - dones) * next_q
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "steps": self.steps,
                    "updates": self.updates, "config": self.config.__dict__,
                    "metadata": metadata or {}}, path)

    def load(self, path: str | Path, load_optimizer: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = int(checkpoint.get("steps", 0)); self.updates = int(checkpoint.get("updates", 0))
        return checkpoint.get("metadata", {})
