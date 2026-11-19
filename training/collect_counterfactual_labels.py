"""Collect action labels from counterfactual rollouts for offline imitation learning."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
from agents.counterfactual import CounterfactualGreedy

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=1)
    parser.add_argument('--items',type=int,default=20)
    parser.add_argument('--rollout-depth',type=int,default=1000)
    parser.add_argument('--scenarios',default='ABC')
    parser.add_argument('--start-seed',type=int,default=900)
    parser.add_argument('--output',default='artifacts/counterfactual_labels.npz')
    args=parser.parse_args()
    states=[]; masks=[]; labels=[]; candidate_scores=[]; scenario_ids=[]; seeds=[]
    scenario_map={'A':0,'B':1,'C':2}
    for scenario in args.scenarios:
        for seed in range(args.start_seed,args.start_seed+args.seeds):
            env=WarehouseEnv(WarehouseConfig(num_items=args.items,scenario=scenario))
            obs,_=env.reset(seed=seed)
            agent=CounterfactualGreedy(rollout_placements=args.rollout_depth,collect_records=True)
            done=False
            while not done:
                mask=obs['action_mask']
                if not mask.any():
                    obs,_,term,trunc,_=env.step(0); done=term or trunc; continue
                action=agent.select_action(obs,mask,env=env)
                obs,_,term,trunc,_=env.step(action); done=term or trunc
            for record in agent.records:
                states.append(record['observation']); masks.append(record['action_mask']); labels.append(record['label'])
                candidate_scores.append(record['candidate_scores']); scenario_ids.append(scenario_map[scenario]); seeds.append(seed)
            metrics=env.get_metrics()
            print(f"scenario={scenario} seed={seed} decisions={len(agent.records)} reshuffles={metrics.get('reshuffles')} retrieved={metrics.get('retrieved')}",flush=True)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,states=np.asarray(states,dtype=np.float32),action_masks=np.asarray(masks,dtype=np.int8),labels=np.asarray(labels,dtype=np.int64),candidate_scores=np.asarray(candidate_scores,dtype=np.float32),scenario_ids=np.asarray(scenario_ids,dtype=np.int8),seeds=np.asarray(seeds,dtype=np.int64))
    print(f"Saved {len(labels)} labeled decisions to {out}")
if __name__=='__main__': main()
