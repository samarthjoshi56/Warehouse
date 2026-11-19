"""Blocker-aware stack placement heuristic (research baseline)."""
from __future__ import annotations
import numpy as np
from .base_baseline import BaselineAgent
from .strategies import _valid


class BlockerAwarePlacement(BaselineAgent):
    """Minimize estimated retrieval blockers, then stack height.

    Unknown retrieval times trigger a height-only fallback.
    """
    def __init__(self, blocker_weight: float = 2.0, height_weight: float = 0.25,
                 temp_pressure_weight: float = 0.15):
        self.blocker_weight = float(blocker_weight)
        self.height_weight = float(height_weight)
        self.temp_pressure_weight = float(temp_pressure_weight)

    @property
    def name(self):
        return "Blocker-Aware"

    def select_action(self, observation, action_mask):
        valid = _valid(action_mask)
        grid = np.asarray(observation["grid"])
        heights = np.asarray(observation["stack_summaries"][:, 0], dtype=float)
        incoming_tau = float(observation["incoming_item"][0])
        temp_pressure = float(observation.get("temp_storage", [0.0])[0])
        unknown = incoming_tau < 0
        scores = []
        for k in valid:
            occupied = np.flatnonzero(grid[k, :, 0] > 0.5)
            h = len(occupied)
            blockers = 0.0
            if not unknown:
                for slot in occupied:
                    tau_existing = float(grid[k, slot, 1])
                    if tau_existing >= 0 and tau_existing < incoming_tau:
                        blockers += 1.0 + 0.25 * (h - int(slot) - 1)
            score = (self.blocker_weight * blockers + self.height_weight * h
                     + self.temp_pressure_weight * temp_pressure * heights[k])
            scores.append((score, h, int(k)))
        return min(scores)[-1]
