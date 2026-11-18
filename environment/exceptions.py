"""
Custom exceptions for the warehouse simulator.

All exceptions raised by the simulator are defined here.
No other module should define simulator exceptions.
"""


class WarehouseError(Exception):
    """Base class for all warehouse simulator exceptions."""


class InvalidActionError(WarehouseError):
    """Raised when the agent selects a masked (invalid) action."""


class NoValidPlacementError(WarehouseError):
    """Raised when no valid stack exists for the incoming item."""


class TemporaryStorageFullError(WarehouseError):
    """Raised when temporary storage is at capacity and cannot accept a blocker."""


class DuplicateItemError(WarehouseError):
    """Raised when an item with the same ID is placed in the warehouse twice."""


class ItemNotFoundError(WarehouseError):
    """Raised when a retrieval target does not exist in the warehouse."""


class ItemConservationError(WarehouseError):
    """Raised when the item conservation invariant is violated."""


class RestoreFailure(WarehouseError):
    """Raised when a blocker cannot be restored to any valid warehouse stack."""


class IncompatibleDestinationError(WarehouseError):
    """Raised when a blocker has no compatible destination during reshuffling."""
