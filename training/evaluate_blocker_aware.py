"""Paired-seed comparison of Blocker-Aware and Improved Greedy baselines."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
from baselines.strategies import ImprovedGreedy
from baselines.blocker_aware import BlockerAwarePlacement


def run(agent, scenario, seed, items):
    env=WarehouseEnv(WarehouseConfig(num_items=items,scenario=scenario))
    obs,_=env.reset(seed=seed); agent.reset(); done=False
    while not done:
        mask=obs['action_mask']
        if not mask.any():
            obs,_,term,trunc,_=env.step(0); done=term or trunc; continue
        action=agent.select_action(obs,mask)
        obs,_,term,trunc,_=env.step(action); done=term or trunc
    return env.get_metrics()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=30)
    p.add_argument('--items',type=int,default=20)
    p.add_argument('--start-seed',type=int,default=1000)
    p.add_argument('--output',default='artifacts/blocker_aware_benchmark.csv')
    a=p.parse_args(); rows=[]
    for sc in 'ABC':
        for seed in range(a.start_seed,a.start_seed+a.seeds):
            for name,agent in [('Improved Greedy',ImprovedGreedy()),('Blocker-Aware',BlockerAwarePlacement())]:
                m=run(agent,sc,seed,a.items)
                rows.append({'scenario':sc,'policy':name,'seed':seed,'items':a.items,
                    'reshuffles':m.get('reshuffles',0),'retrieved':m.get('retrieved',0),
                    'failed':m.get('failed',0),'unplaced':m.get('unplaced',0),
                    'invalid_actions':m.get('invalid_actions',0)})
                print(f"{sc} {name} seed={seed} reshuffles={m.get('reshuffles')}",flush=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('\\nSUMMARY (mean reshuffles)')
    for sc in 'ABC':
        for name in ['Improved Greedy','Blocker-Aware']:
            vals=[r['reshuffles'] for r in rows if r['scenario']==sc and r['policy']==name]
            print(f'{sc} {name}: {np.mean(vals):.3f}')
    for sc in 'ABC':
        ig={r['seed']:r['reshuffles'] for r in rows if r['scenario']==sc and r['policy']=='Improved Greedy'}
        ba={r['seed']:r['reshuffles'] for r in rows if r['scenario']==sc and r['policy']=='Blocker-Aware'}
        diffs=[ba[k]-ig[k] for k in ig]
        print(f'{sc} paired delta (Blocker-Aware - IG): mean={np.mean(diffs):.3f}, wins={sum(x<0 for x in diffs)}, ties={sum(x==0 for x in diffs)}, losses={sum(x>0 for x in diffs)}')
if __name__=='__main__': main()
