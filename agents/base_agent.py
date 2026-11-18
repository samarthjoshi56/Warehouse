"""
Abstract agent interface for RL agents.

Phase 1: Interface definition only.
Implementation deferred to Phase 2.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractAgent(ABC):
    """
    Base interface for all RL agents.

    Baselines and RL agents share this interface so that the
    evaluation runner can treat them interchangeably.
    """

    @abstractmethod
    def select_action(
        self,
        observation: dict,
        action_mask: np.ndarray,
    ) -> int:
        """
        Select a stack index for the incoming item.

        Args:
            observation: Agent-visible state dict from WarehouseEnv.
            action_mask: Boolean array of shape (K,); True = valid stack.

        Returns:
            Stack index k in {0, ..., K-1}.

        Raises:
            InvalidActionError: If a masked action is selected.
        """
        ...

    def reset(self) -> None:
        """Called at the start of each episode. Override if stateful."""
        pass

    def update(self, *args, **kwargs) -> None:
        """Called after each step for online learning. Override if needed."""
        pass
