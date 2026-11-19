"""Small reproducible benchmark for counterfactual placement rollouts."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
from baselines.strategies import ImprovedGreedy
from agents.counterfactual import CounterfactualGreedy

def run(agent, scenario, seed, n_items):
    env=WarehouseEnv(WarehouseConfig(num_items=n_items,scenario=scenario))
    obs,_=env.reset(seed=seed); agent.reset(); done=False
    while not done:
        mask=obs['action_mask']
        if not mask.any():
            obs,_,term,trunc,_=env.step(0); done=term or trunc; continue
        action=agent.select_action(obs,mask,env=env) if isinstance(agent,CounterfactualGreedy) else agent.select_action(obs,mask)
        obs,_,term,trunc,_=env.step(action); done=term or trunc
    return env.get_metrics()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',type=int,default=3); p.add_argument('--items',type=int,default=30); args=p.parse_args()
    rows=[]
    for sc in 'ABC':
        for name,agent in [('Improved Greedy',ImprovedGreedy()),('Counterfactual',CounterfactualGreedy())]:
            for seed in range(700,700+args.seeds):
                m=run(agent,sc,seed,args.items)
                rows.append({'scenario':sc,'policy':name,'seed':seed,'items':args.items,'reshuffles':m.get('reshuffles',0),'retrieved':m.get('retrieved',0),'failed':m.get('failed',0),'unplaced':m.get('unplaced',0),'invalid_actions':m.get('invalid_actions',0)})
                print(f"{sc} {name} seed={seed} reshuffles={m.get('reshuffles')}",flush=True)
    out=Path(__file__).resolve().parents[1]/'artifacts'/'counterfactual_benchmark.csv'
    out.parent.mkdir(exist_ok=True)
    with out.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for sc in 'ABC':
        for name in ['Improved Greedy','Counterfactual']:
            vals=[r['reshuffles'] for r in rows if r['scenario']==sc and r['policy']==name]
            print(f"summary {sc} {name} mean={np.mean(vals):.3f}")
if __name__=='__main__': main()
