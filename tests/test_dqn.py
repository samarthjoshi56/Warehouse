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
