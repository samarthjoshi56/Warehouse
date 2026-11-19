import numpy as np
from agents.dqn import DQNAgent, DQNConfig, flatten_observation
from training.replay_buffer import ReplayBuffer
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv


def test_dqn_never_selects_masked_action():
    env = WarehouseEnv(WarehouseConfig(num_stacks=3, stack_height=2, num_items=2, seed=3))
    obs, _ = env.reset(seed=3)
    agent = DQNAgent(flatten_observation(obs).size, env.action_space.n,
                     DQNConfig(epsilon_start=0.0, epsilon_end=0.0), seed=3)
    for _ in range(10):
        action = agent.select_action(obs, explore=False)
        assert obs["action_mask"][action] == 1


def test_replay_buffer_samples_correct_shapes():
    buffer = ReplayBuffer(8, observation_dim=5, action_dim=2, seed=1)
    for i in range(4):
        buffer.add(np.ones(5) * i, i % 2, float(i), np.ones(5) * (i + 1), False, [True, False])
    batch = buffer.sample(3)
    assert batch["states"].shape == (3, 5)
    assert batch["next_masks"].shape == (3, 2)


def test_retrieval_reward_includes_reshuffle_cost_and_delay():
    cfg = WarehouseConfig(num_stacks=1, stack_height=2, temp_capacity=1,
                          num_items=2, arrival_interval=1, min_dwell_time=100,
                          max_dwell_time=100, max_episode_steps=500, scenario="C")
    env = WarehouseEnv(cfg)
    obs, _ = env.reset(seed=7)
    obs, _, term, trunc, _ = env.step(0)
    assert not term and not trunc
    obs, reward, term, trunc, info = env.step(0)
    assert term or trunc
    assert info["reshuffles"] == 1
    assert info["retrieved"] == 2
    # Reward must include +2 per retrieval and -1 for the blocker reshuffle.
    assert reward > 0.0


def test_dqn_supports_shared_agent_interface():
    env = WarehouseEnv(WarehouseConfig(num_stacks=3, stack_height=2, num_items=2, seed=5))
    obs, _ = env.reset(seed=5)
    agent = DQNAgent(flatten_observation(obs).size, env.action_space.n, seed=5)
    action = agent.select_action(obs, obs["action_mask"], explore=False)
    assert obs["action_mask"][action] == 1


def test_episode_runner_accepts_dqn_agent():
    from evaluation.runner import EpisodeRunner
    config = WarehouseConfig(num_stacks=3, stack_height=2, num_items=4,
                             max_episode_steps=200, min_dwell_time=5, max_dwell_time=8)
    probe = WarehouseEnv(config)
    obs, _ = probe.reset(seed=11)
    agent = DQNAgent(flatten_observation(obs).size, probe.action_space.n, seed=11)
    runner = EpisodeRunner(lambda: WarehouseEnv(config))
    result = runner.run(agent, [11])
    assert len(result) == 1
    assert result[0]["seed"] == 11


def test_observation_v2_shapes_and_full_compatibility_vector():
    cfg = WarehouseConfig(num_stacks=4, stack_height=3, num_groups=3,
                         stack_groups=(0, 1, 2, 0), num_items=6,
                         observation_version=2, seed=12)
    env = WarehouseEnv(cfg)
    obs, _ = env.reset(seed=12)
    assert obs["grid"].shape == (4, 3, 8)
    assert obs["incoming_item"].shape == (6,)
    assert obs["action_mask"].shape == (4,)
    item = env.current_item
    if item is not None:
        np.testing.assert_array_equal(obs["incoming_item"][3:], 
                                   [float(g in item.compatible_groups) for g in range(3)])
        assert obs["incoming_item"][2] == 1.0


def test_observation_v2_stack_group_is_one_hot():
    cfg = WarehouseConfig(num_stacks=3, stack_height=2, num_groups=3,
                         stack_groups=(0, 1, 2), num_items=4,
                         observation_version=2, seed=8)
    env = WarehouseEnv(cfg)
    obs, _ = env.reset(seed=8)
    # Place one item in stack 1 and inspect the occupied slot.
    if obs["action_mask"][1]:
        obs, _, _, _, _ = env.step(1)
        occupied = obs["grid"][1, 0]
        assert occupied[0] == 1.0
        assert occupied[5:8].sum() == 1.0
        assert occupied[6] == 1.0


def test_observation_v1_remains_default_and_compatible():
    cfg = WarehouseConfig(num_stacks=3, stack_height=2, num_items=3, seed=2)
    env = WarehouseEnv(cfg)
    obs, _ = env.reset(seed=2)
    assert cfg.observation_version == 1
    assert obs["grid"].shape == (3, 2, 6)
    assert obs["incoming_item"].shape == (4,)
