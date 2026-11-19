"""Counterfactual one-step placement scorer using cloned simulator rollouts."""
from __future__ import annotations
import copy
import numpy as np
from baselines.strategies import ImprovedGreedy

class CounterfactualGreedy:
    """Try each valid placement, then finish the episode with Improved Greedy.


    This is a model-based decision aid, not a learned policy. It can be costly:
    each decision simulates one full remaining episode per valid action.
    """
    def __init__(self, rollout_agent=None, max_actions=None):
        self.rollout_agent = rollout_agent or ImprovedGreedy()
        self.max_actions = max_actions
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
        results=[]
        for action in valid:
            clone=copy.deepcopy(env)
            clone_obs, _, terminated, truncated, _ = clone.step(int(action))
            self.rollout_agent.reset()
            while not (terminated or truncated):
                mask=clone_obs["action_mask"]
                if not mask.any():
                    clone_obs,_,terminated,truncated,_=clone.step(0)
                    continue
                next_action=self.rollout_agent.select_action(clone_obs,mask)
                clone_obs,_,terminated,truncated,_=clone.step(next_action)
            metrics=clone.get_metrics()
            self.evaluations += 1
            results.append((float(metrics.get("reshuffles",0)), float(metrics.get("failed",0)), float(metrics.get("unplaced",0)), int(action)))
        return min(results)[-1]
