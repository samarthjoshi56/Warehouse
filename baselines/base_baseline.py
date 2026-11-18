"""
Abstract baseline agent interface.

All five baselines implement this interface.
Baselines must only receive information available to the RL agent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaselineAgent(ABC):
    """
    Base class for all deterministic baseline placement algorithms.

    Baselines implement the same interface as AbstractAgent so they
    can be evaluated by the same EpisodeRunner.
    """

    @abstractmethod
    def select_action(
        self,
        observation: dict,
        action_mask: np.ndarray,
    ) -> int:
        """
        Select a valid stack index for the incoming item.

        Args:
            observation: Agent-visible state dict from WarehouseEnv.
            action_mask: Boolean array of shape (K,); True = valid stack.

        Returns:
            Stack index k in {0, ..., K-1}.
        """
        ...

    def reset(self) -> None:
        """Called at the start of each episode."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for reporting."""
        ...
