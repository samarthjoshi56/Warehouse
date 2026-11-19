import numpy as np
from pathlib import Path
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
from baselines.strategies import ImprovedGreedy
from agents.dqn import flatten_observation
from agents.hybrid import GuardedHybridAgent
CHECKPOINT = Path(__file__).resolve().parents[1] / "artifacts" / "dqn_warehouse.pt"
def test_hybrid_high_threshold_never_overrides_heuristic():
    env=WarehouseEnv(WarehouseConfig(num_items=12,scenario="B")); obs,_=env.reset(seed=42)
    hybrid=GuardedHybridAgent(flatten_observation(obs).size,env.action_space.n,str(CHECKPOINT),1e9,seed=42)
    heuristic=ImprovedGreedy()
    for _ in range(10):
        mask=obs["action_mask"]
        if not mask.any(): break
        assert hybrid.select_action(obs,mask)==heuristic.select_action(obs,mask)
        obs,_,term,trunc,_=env.step(heuristic.select_action(obs,mask))
        if term or trunc: break
def test_hybrid_action_always_valid():
    env=WarehouseEnv(WarehouseConfig(num_items=15,scenario="C")); obs,_=env.reset(seed=123)
    hybrid=GuardedHybridAgent(flatten_observation(obs).size,env.action_space.n,str(CHECKPOINT),0,seed=123)
    done=False
    while not done:
        mask=obs["action_mask"]
        if not mask.any(): break
        action=hybrid.select_action(obs,mask); assert mask[action]==1
        obs,_,term,trunc,_=env.step(action); done=term or trunc
