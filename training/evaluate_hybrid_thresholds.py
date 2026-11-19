"""Sweep Q-margin thresholds for the heuristic-guarded hybrid policy."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
from baselines.strategies import ImprovedGreedy
from agents.dqn import DQNAgent,DQNConfig,flatten_observation
from agents.hybrid import GuardedHybridAgent
ROOT=Path(__file__).resolve().parents[1]; CKPT=ROOT/"artifacts"/"dqn_warehouse.pt"
def run_episode(agent,scenario,seed,num_items=100):
    env=WarehouseEnv(WarehouseConfig(num_items=num_items,scenario=scenario)); obs,_=env.reset(seed=seed); agent.reset(); done=False
    while not done:
        mask=obs["action_mask"]
        if not mask.any():
            obs,_,terminated,truncated,_=env.step(0); done=terminated or truncated; continue
        action=agent.select_action(obs,mask); obs,_,terminated,truncated,_=env.step(action); done=terminated or truncated
    return env.get_metrics()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--seeds",type=int,default=15); p.add_argument("--items",type=int,default=100); p.add_argument("--thresholds",default="0,0.25,0.5,1,2,5,10,1000"); args=p.parse_args()
    thresholds=[float(x) for x in args.thresholds.split(",")]; seeds=list(range(500,500+args.seeds)); records=[]
    for scenario in "ABC":
        env=WarehouseEnv(WarehouseConfig(num_items=args.items,scenario=scenario)); obs,_=env.reset(seed=seeds[0]); dim,actions=flatten_observation(obs).size,env.action_space.n
        policies=[("Improved Greedy",ImprovedGreedy())]; dqn=DQNAgent(dim,actions,DQNConfig(),seed=0); dqn.load(CKPT); policies.append(("DQN",dqn))
        policies.extend((f"Hybrid margin {t:g}",GuardedHybridAgent(dim,actions,str(CKPT),t,seed=0)) for t in thresholds)
        for name,policy in policies:
            for seed in seeds:
                m=run_episode(policy,scenario,seed,args.items)
                records.append({"scenario":scenario,"policy":name,"seed":seed,"items":args.items,"reshuffles":m.get("reshuffles",0),"retrieved":m.get("retrieved",0),"failed":m.get("failed",0),"unplaced":m.get("unplaced",0),"invalid_actions":m.get("invalid_actions",0),"override_rate":getattr(policy,"override_rate",None)})
            vals=[r["reshuffles"] for r in records if r["scenario"]==scenario and r["policy"]==name]
            print(f"scenario={scenario} policy={name} mean={np.mean(vals):.3f} sd={np.std(vals,ddof=1):.3f}",flush=True)
    out=ROOT/"artifacts"/"hybrid_threshold_sweep.csv"
    with out.open("w",newline="") as f: writer=csv.DictWriter(f,fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    print(f"Saved {len(records)} rows to {out}")
if __name__=="__main__": main()
