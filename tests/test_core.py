import numpy as np
import pytest

from environment import WarehouseConfig, WarehouseEnv
from environment.item import Item
from environment.stack import Stack
from environment.temp_storage import TempStorage
from environment.metrics import MetricsTracker
from environment.retrieval import RetrievalEngine
from environment.exceptions import InvalidActionError, TemporaryStorageFullError


def test_stack_lifo_and_capacity():
    stack = Stack(0, 0, 2)
    a = Item("A", 0, 1, 1)
    b = Item("B", 0, 2, 2)
    stack.push(a); stack.push(b)
    assert stack.pop().id == "B"
    assert stack.pop().id == "A"
    with pytest.raises(IndexError): stack.pop()


def test_temp_storage_capacity():
    temp = TempStorage(1)
    temp.add(Item("A", 0, 1, 1))
    with pytest.raises(TemporaryStorageFullError): temp.add(Item("B", 0, 2, 2))
    assert temp.remove("A").id == "A"


def test_observation_shapes_and_compact_vector():
    env = WarehouseEnv(WarehouseConfig(num_stacks=3, stack_height=2, temp_capacity=2, num_items=3, max_episode_steps=100, min_dwell_time=10, max_dwell_time=10))
    obs, _ = env.reset(seed=7)
    assert obs["grid"].shape == (3, 2, 6)
    assert obs["incoming_item"].shape == (4,)
    assert obs["temp_storage"].shape == (3,)
    assert obs["stack_summaries"].shape == (3, 3)
    assert obs["schedule_lookahead"].shape == (20, 4)
    assert obs["action_mask"].shape == (3,)
    assert env.state.to_compact_vector().shape == (3 * 3 + 1 + 4,)


def test_invalid_action_is_rejected():
    env = WarehouseEnv(WarehouseConfig(num_stacks=2, stack_height=1, temp_capacity=1, num_items=2, max_episode_steps=100, min_dwell_time=20, max_dwell_time=20))
    obs, _ = env.reset(seed=1)
    with pytest.raises(InvalidActionError): env.step(9)


def test_episode_runs_and_conserves_items():
    cfg = WarehouseConfig(num_stacks=3, stack_height=3, temp_capacity=4, num_items=8, max_episode_steps=500, min_dwell_time=5, max_dwell_time=15, scenario="C")
    env = WarehouseEnv(cfg)
    obs, _ = env.reset(seed=42)
    terminated = truncated = False
    while not (terminated or truncated):
        mask = obs["action_mask"]
        if mask.any():
            action = int(np.flatnonzero(mask)[0])
        else:
            # The environment should normally auto-discard only after the arrival is advanced.
            action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        env._assert_conservation()
    assert env.metrics.data.placements + env.metrics.data.unplaced == cfg.num_items
    assert env.metrics.data.retrieved <= cfg.num_items


def test_blocker_retrieval_counts_each_move():
    stacks = [Stack(0, 0, 5), Stack(1, 0, 5)]
    temp = TempStorage(4)
    metrics = MetricsTracker()
    engine = RetrievalEngine(stacks, temp, metrics)
    a = Item("A", 0, 10, 10)
    b = Item("B", 0, 20, 20)
    c = Item("C", 0, 30, 30)
    stacks[0].push(a); stacks[0].push(b); stacks[0].push(c)
    reshuffles, duration = engine.retrieve(a, 10)
    assert reshuffles == 2
    assert duration == 5
    assert metrics.data.reshuffles == 2
    assert metrics.data.retrieved == 1
    assert temp.count <= 4
