"""Heuristic-guarded DQN policy for experimental evaluation."""
from __future__ import annotations
import numpy as np
import torch
from .dqn import DQNAgent, flatten_observation
from baselines.strategies import ImprovedGreedy

class GuardedHybridAgent:
    """Improved Greedy fallback; DQN overrides only when its Q-margin is high.

    Q-margin is a heuristic confidence proxy, not a calibrated probability.
    """
    def __init__(self, observation_dim: int, action_dim: int, checkpoint: str,
                 q_margin_threshold: float = 0.0, seed: int = 0, device: str | None = None):
        self.heuristic = ImprovedGreedy()
        self.dqn = DQNAgent(observation_dim, action_dim, device=device, seed=seed)
        self.dqn.load(checkpoint)
        self.threshold = float(q_margin_threshold)
        self.overrides = 0
        self.decisions = 0
        self.margin_sum = 0.0
    def reset(self):
        self.heuristic.reset(); self.dqn.reset()
        self.overrides = 0; self.decisions = 0; self.margin_sum = 0.0
    def select_action(self, observation, action_mask):
        mask = np.asarray(action_mask, dtype=bool)
        valid = np.flatnonzero(mask)
        if not len(valid): raise ValueError("No valid action is available")
        fallback = self.heuristic.select_action(observation, mask)
        state = torch.as_tensor(flatten_observation(observation), dtype=torch.float32,
                              device=self.dqn.device).unsqueeze(0)
        with torch.no_grad():
            q = self.dqn.online(state).squeeze(0)
            q = q.masked_fill(torch.as_tensor(~mask, dtype=torch.bool, device=self.dqn.device), -torch.inf)
        sorted_q = torch.sort(q[torch.as_tensor(valid, device=self.dqn.device)], descending=True).values
        margin = float((sorted_q[0] - sorted_q[1]).item()) if len(sorted_q) > 1 else float("inf")
        learned_action = int(torch.argmax(q).item())
        self.decisions += 1
        self.margin_sum += margin if np.isfinite(margin) else 0.0
        if learned_action != fallback and margin >= self.threshold:
            self.overrides += 1
            return learned_action
        return fallback
    @property
    def override_rate(self):
        return self.overrides / self.decisions if self.decisions else 0.0
