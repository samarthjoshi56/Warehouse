"""Evaluate a counterfactual-label clone on new seeds."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
from agents.counterfactual_clone import CounterfactualClone
from baselines.strategies import ImprovedGreedy
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
def run(agent,scenario,seed,n):
 env=WarehouseEnv(WarehouseConfig(num_items=n,scenario=scenario)); obs,_=env.reset(seed=seed); agent.reset(); done=False
 while not done:
  mask=obs['action_mask']
  if not mask.any(): obs,_,term,trunc,_=env.step(0); done=term or trunc; continue
  action=agent.select_action(obs,mask); obs,_,term,trunc,_=env.step(action); done=term or trunc
 return env.get_metrics()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',default='artifacts/counterfactual_bc.pt'); p.add_argument('--items',type=int,default=20); p.add_argument('--seeds',default='901,902,903'); args=p.parse_args(); seeds=[int(x) for x in args.seeds.split(',')]; rows=[]; clone=CounterfactualClone(args.checkpoint)
 for sc in 'ABC':
  for name,agent in [('Improved Greedy',ImprovedGreedy()),('Counterfactual Clone',clone)]:
   for seed in seeds:
    m=run(agent,sc,seed,args.items); rows.append({'scenario':sc,'policy':name,'seed':seed,'items':args.items,'reshuffles':m.get('reshuffles',0),'retrieved':m.get('retrieved',0),'failed':m.get('failed',0),'unplaced':m.get('unplaced',0),'invalid_actions':m.get('invalid_actions',0)})
    print(f"{sc} {name} seed={seed} reshuffles={m.get('reshuffles')}")
 out=Path('artifacts/counterfactual_clone_evaluation.csv')
 with out.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 for sc in 'ABC':
  for name in ['Improved Greedy','Counterfactual Clone']:
   vals=[r['reshuffles'] for r in rows if r['scenario']==sc and r['policy']==name]; print(f"summary {sc} {name} mean={sum(vals)/len(vals):.3f}")
if __name__=='__main__': main()
