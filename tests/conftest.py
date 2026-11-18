"""
Shared pytest fixtures for all test tiers.
Implementations populated in Phase 2 when WarehouseEnv is built.
"""
import pytest


# ---------------------------------------------------------------------------
# Fixtures — to be completed in Phase 2
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_warehouse():
    """Return an empty WarehouseEnv with default parameters."""
    # from environment.warehouse import WarehouseEnv
    # return WarehouseEnv(num_stacks=10, stack_height=5, temp_capacity=10, seed=42)
    pytest.skip("WarehouseEnv not yet implemented — Phase 2")


@pytest.fixture
def full_warehouse():
    """Return a WarehouseEnv with all slots occupied."""
    pytest.skip("WarehouseEnv not yet implemented — Phase 2")


@pytest.fixture
def sample_item():
    """Return a sample Item for use in tests."""
    # Item model is introduced in Phase 2.
    pytest.skip("Item not yet implemented — Phase 2")
