"""Counterfactual bounded-rollout placement scorer using cloned simulator states."""
from __future__ import annotations
import copy
import numpy as np
from baselines.strategies import ImprovedGreedy


class CounterfactualGreedy:
    """Test each valid action, then roll out a bounded number of placements.

    This model-based agent requires access to the live simulator and is not an
    observation-only deployment policy. The rollout horizon controls compute cost.
    """
    def __init__(self, rollout_agent=None, max_actions=None, rollout_placements=12):
        if rollout_placements < 1:
            raise ValueError("rollout_placements must be at least 1")
        self.rollout_agent = rollout_agent or ImprovedGreedy()
        self.max_actions = max_actions
        self.rollout_placements = int(rollout_placements)
        self.evaluations = 0
        self.reset()

    def reset(self):
        self.evaluations = 0
        if hasattr(self.rollout_agent, "reset"):
            self.rollout_agent.reset()

    def select_action(self, observation, action_mask, env=None):
        if env is None:
            raise ValueError("CounterfactualGreedy requires the live environment instance")
        valid = np.flatnonzero(np.asarray(action_mask, dtype=np.int8))
        if not len(valid):
            raise ValueError("No valid action available")
        if self.max_actions is not None and len(valid) > self.max_actions:
            valid = valid[:self.max_actions]
        results = []
        for action in valid:
            clone = copy.deepcopy(env)
            clone_obs, _, terminated, truncated, _ = clone.step(int(action))
            placements = 1
            self.rollout_agent.reset()
            while not (terminated or truncated) and placements < self.rollout_placements:
                mask = clone_obs["action_mask"]
                if not mask.any():
                    clone_obs, _, terminated, truncated, _ = clone.step(0)
                    continue
                next_action = self.rollout_agent.select_action(clone_obs, mask)
                clone_obs, _, terminated, truncated, _ = clone.step(next_action)
                placements += 1
            metrics = clone.get_metrics()
            self.evaluations += 1
            results.append((float(metrics.get("reshuffles", 0)),
                           float(metrics.get("failed", 0)),
                           float(metrics.get("unplaced", 0)), int(action)))
        return min(results)[-1]
