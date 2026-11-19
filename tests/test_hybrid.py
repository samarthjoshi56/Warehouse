import numpy as np
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
from baselines.strategies import ImprovedGreedy
from agents.dqn import DQNAgent, DQNConfig, flatten_observation
from agents.hybrid import GuardedHybridAgent


def make_hybrid(env, obs, threshold, seed, checkpoint_path):
    agent = DQNAgent(flatten_observation(obs).size, env.action_space.n, DQNConfig(), seed=seed)
    agent.save(checkpoint_path, metadata={"test": True})
    return GuardedHybridAgent(flatten_observation(obs).size, env.action_space.n,
                             str(checkpoint_path), threshold, seed=seed)


def test_hybrid_high_threshold_never_overrides_heuristic(tmp_path):
    env = WarehouseEnv(WarehouseConfig(num_items=12, scenario="B"))
    obs, _ = env.reset(seed=42)
    hybrid = make_hybrid(env, obs, 1e9, 42, tmp_path / "synthetic.pt")
    heuristic = ImprovedGreedy()
    for _ in range(10):
        mask = obs["action_mask"]
        if not mask.any():
            break
        assert hybrid.select_action(obs, mask) == heuristic.select_action(obs, mask)
        obs, _, term, trunc, _ = env.step(heuristic.select_action(obs, mask))
        if term or trunc:
            break


def test_hybrid_action_always_valid(tmp_path):
    env = WarehouseEnv(WarehouseConfig(num_items=15, scenario="C"))
    obs, _ = env.reset(seed=123)
    hybrid = make_hybrid(env, obs, 0, 123, tmp_path / "synthetic.pt")
    done = False
    while not done:
        mask = obs["action_mask"]
        if not mask.any():
            break
        action = hybrid.select_action(obs, mask)
        assert mask[action] == 1
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
