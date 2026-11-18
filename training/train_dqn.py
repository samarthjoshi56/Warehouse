"""Train a masked DQN policy on reproducible warehouse episodes."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from agents.dqn import DQNAgent, DQNConfig, flatten_observation
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
from training.replay_buffer import ReplayBuffer

def train(episodes: int = 50, seed_start: int = 0, scenario: str = "B",
          output: str = "artifacts/dqn_warehouse.pt", num_items: int = 100,
          config: DQNConfig | None = None, resume: bool = False) -> list[dict]:
    dqn_cfg = config or DQNConfig()
    env = WarehouseEnv(WarehouseConfig(num_items=num_items, scenario=scenario))
    obs, _ = env.reset(seed=seed_start)
    obs_dim = flatten_observation(obs).size
    agent = DQNAgent(obs_dim, env.action_space.n, dqn_cfg, seed=seed_start)
    output_path = Path(output); csv_path = output_path.with_suffix(".csv")
    start_episode = 0; history = []
    if resume and output_path.exists():
        metadata = agent.load(output_path, load_optimizer=True)
        start_episode = int(metadata.get("episodes_completed", 0))
        dqn_cfg = agent.config
        if csv_path.exists():
            with csv_path.open(newline="") as f:
                history = list(csv.DictReader(f))
    replay = ReplayBuffer(dqn_cfg.replay_capacity, obs_dim, env.action_space.n, seed_start + start_episode)
    for episode in range(start_episode, episodes):
        obs, _ = env.reset(seed=seed_start + episode)
        state = flatten_observation(obs); total_reward = 0.0; losses = []; done = False
        while not done:
            if not np.asarray(obs["action_mask"], dtype=bool).any():
                break
            action = agent.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = flatten_observation(next_obs)
            replay.add(state, action, reward, next_state, terminated or truncated,
                       np.asarray(next_obs["action_mask"], dtype=bool))
            total_reward += reward; state, obs = next_state, next_obs
            done = terminated or truncated
            if len(replay) >= max(dqn_cfg.learning_starts, dqn_cfg.batch_size) and agent.steps % dqn_cfg.train_frequency == 0:
                losses.append(agent.update(replay.sample(dqn_cfg.batch_size)))
        row = {"episode": episode + 1, "seed": seed_start + episode,
               "return": total_reward, "mean_loss": float(np.mean(losses)) if losses else None,
               **env.get_metrics()}
        history.append(row)
        agent.save(output, {"scenario": scenario, "episodes_completed": episode + 1,
                            "requested_episodes": episodes, "seed_start": seed_start})
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        print(json.dumps(row), flush=True)
    agent.save(output, {"scenario": scenario, "episodes_completed": episodes,
                        "requested_episodes": episodes, "seed_start": seed_start})
    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--scenario", choices=["A", "B", "C"], default="B")
    parser.add_argument("--num-items", type=int, default=100)
    parser.add_argument("--output", default="artifacts/dqn_warehouse.pt")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.episodes, args.seed_start, args.scenario, args.output, args.num_items, resume=args.resume)
if __name__ == "__main__":
    main()
