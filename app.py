"""Gradio demo for the Smart Warehouse Storage Optimisation Agent."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import gradio as gr

from agents.dqn import DQNAgent, DQNConfig, flatten_observation
from baselines.strategies import RandomPlacement, FirstFit, GreedyMinHeight, EarliestRetrievalFirst, ImprovedGreedy
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv

CHECKPOINT = Path("artifacts/dqn_warehouse.pt")
BASELINE_NAMES = ["Random", "First Fit", "Greedy (Min Height)", "Earliest Retrieval First", "Improved Greedy"]
POLICY_NAMES = BASELINE_NAMES + (["Trained DQN"] if CHECKPOINT.exists() else [])


def run_demo(policy_name: str, scenario: str, num_items: int, seed: int):
    config = WarehouseConfig(num_items=int(num_items), scenario=scenario)
    env = WarehouseEnv(config)
    obs, _ = env.reset(seed=int(seed))
    if policy_name == "Random":
        policy = RandomPlacement(seed)
    elif policy_name == "First Fit":
        policy = FirstFit()
    elif policy_name == "Greedy (Min Height)":
        policy = GreedyMinHeight()
    elif policy_name == "Earliest Retrieval First":
        policy = EarliestRetrievalFirst()
    elif policy_name == "Improved Greedy":
        policy = ImprovedGreedy()
    elif policy_name == "Trained DQN":
        if not CHECKPOINT.exists():
            raise gr.Error("No trained checkpoint found. Add artifacts/dqn_warehouse.pt to the Space.")
        policy = DQNAgent(flatten_observation(obs).size, env.action_space.n, DQNConfig())
        policy.load(CHECKPOINT)
    else:
        raise gr.Error("Unknown policy selected")

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        mask = obs["action_mask"]
        if not np.asarray(mask).any():
            break
        if policy_name == "Trained DQN":
            action = policy.select_action(obs, explore=False)
        else:
            action = policy.select_action(obs, mask)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    metrics = env.get_metrics()
    metrics["episode_return"] = round(total_reward, 4)
    metrics["agent_decisions"] = steps
    return env.render(), metrics


with gr.Blocks(title="Smart Warehouse RL Agent") as demo:
    gr.Markdown("# Smart Warehouse Storage Optimisation Agent")
    gr.Markdown("Compare warehouse placement strategies. The primary metric is the number of reshuffling operations.")
    with gr.Row():
        policy = gr.Dropdown(POLICY_NAMES, value=POLICY_NAMES[0], label="Placement policy")
        scenario = gr.Dropdown(["A", "B", "C"], value="B", label="Information scenario")
    with gr.Row():
        num_items = gr.Slider(10, 100, value=40, step=5, label="Items per episode")
        seed = gr.Number(value=200, precision=0, label="Episode seed")
    run = gr.Button("Run episode", variant="primary")
    with gr.Row():
        warehouse = gr.Textbox(label="Final warehouse state", lines=18, max_lines=24)
        metrics = gr.JSON(label="Episode metrics")
    run.click(run_demo, inputs=[policy, scenario, num_items, seed], outputs=[warehouse, metrics])
    gr.Markdown("**Note:** DQN is available only when a trained checkpoint is included. Scenario A/B are partially observable. Scenario C uses a finite schedule lookahead and is an approximation to full scheduler observability.")

if __name__ == "__main__":
    demo.launch()
