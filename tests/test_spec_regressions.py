"""Regression tests mapped to the Phase 1.3 formal test specification."""
import numpy as np
import pytest

from environment import WarehouseConfig, WarehouseEnv
from environment.item import Item
from environment.stack import Stack
from environment.temp_storage import TempStorage
from environment.metrics import MetricsTracker
from environment.retrieval import RetrievalEngine
from environment.exceptions import NoValidPlacementError
from baselines import ImprovedGreedy


def test_ut16_actual_retrieval_time_does_not_change_scenario_a_observation():
    cfg = WarehouseConfig(
        num_stacks=3, stack_height=2, temp_capacity=2, num_items=5,
        max_episode_steps=200, min_dwell_time=10, max_dwell_time=10,
        scenario="A",
    )
    env = WarehouseEnv(cfg)
    obs_before, _ = env.reset(seed=17)
    for index, item in enumerate(env.items.values()):
        item.actual_retrieval_time += 37 + index
    obs_after = env._observation()
    for key in obs_before:
        np.testing.assert_array_equal(obs_before[key], obs_after[key], err_msg=key)


def test_ut17_urgency_key_is_a_total_deterministic_order():
    items = [
        Item("ITEM_B", 0, 20, 20, priority=1),
        Item("ITEM_A", 0, 20, 20, priority=2),
        Item("ITEM_C", 0, 20, 20, priority=2),
    ]
    assert [x.id for x in sorted(items, key=lambda x: x.urgency_key)] == [
        "ITEM_B", "ITEM_A", "ITEM_C"
    ]


def test_ut18_improved_greedy_handles_empty_and_missing_estimate():
    agent = ImprovedGreedy()
    obs = {
        "grid": np.zeros((3, 5, 6), dtype=np.float32),
        "incoming_item": np.asarray([-1.0, 0.4, 0.0, 1.0], dtype=np.float32),
        "stack_summaries": np.zeros((3, 3), dtype=np.float32),
    }
    assert agent.select_action(obs, np.asarray([1, 1, 1], dtype=np.int8)) == 0
    with pytest.raises(NoValidPlacementError):
        agent.select_action(obs, np.asarray([0, 0, 0], dtype=np.int8))


def test_ut19_default_compact_observation_is_exactly_35_values():
    env = WarehouseEnv(WarehouseConfig(num_items=10))
    env.reset(seed=23)
    vector = env.state.to_compact_vector()
    assert vector.shape == (35,)
    assert np.isfinite(vector).all()
    assert np.all(vector >= -1.0) and np.all(vector <= 1.0)


def test_ut20_restoration_reverses_full_urgency_key_including_item_id():
    stacks = [Stack(0, 0, 5)]
    temp = TempStorage(3)
    engine = RetrievalEngine(stacks, temp, MetricsTracker())
    for item_id in ("ITEM_02", "ITEM_03", "ITEM_01"):
        temp.add(Item(item_id, 0, 20, 20, priority=1))
    assert engine.restore() == 3
    assert [item.id for item in stacks[0].items] == [
        "ITEM_03", "ITEM_02", "ITEM_01"
    ]
    assert temp.count == 0


def test_scenario_a_observation_does_not_expose_future_schedule():
    cfg = WarehouseConfig(
        num_stacks=3, stack_height=2, temp_capacity=2, num_items=6,
        max_episode_steps=200, min_dwell_time=10, max_dwell_time=10,
        scenario="A",
    )
    env = WarehouseEnv(cfg)
    obs, _ = env.reset(seed=31)
    assert not np.any(obs["schedule_lookahead"])


@pytest.mark.parametrize("scenario", ["A", "B", "C"])
def test_it07_full_100_item_episode_completes_with_valid_metrics(scenario):
    cfg = WarehouseConfig(num_items=100, scenario=scenario, max_episode_steps=4000)
    env = WarehouseEnv(cfg)
    observation, _ = env.reset(seed=42)
    agent = ImprovedGreedy()
    terminated = truncated = False
    while not (terminated or truncated):
        mask = observation["action_mask"]
        action = agent.select_action(observation, mask) if mask.any() else 0
        observation, _, terminated, truncated, _ = env.step(action)
        env._assert_conservation()

    metrics = env.get_metrics()
    assert terminated and not truncated
    assert metrics["placements"] == 100
    assert metrics["retrieved"] == 100
    assert metrics["failed"] == 0
    assert metrics["unplaced"] == 0
    assert np.isfinite(list(metrics.values())).all()
