"""Paired experiment for comparing DQN Observation V1 and V2."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np
from agents.dqn import DQNAgent, DQNConfig, flatten_observation
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv

METRICS = ("reshuffles", "retrieved", "failed", "unplaced", "total_retrieval_delay", "return")

def evaluate_checkpoint(checkpoint: Path, scenario: str, observation_version: int, seeds: list[int]):
    env = WarehouseEnv(WarehouseConfig(scenario=scenario, observation_version=observation_version))
    obs, _ = env.reset(seed=seeds[0])
    agent = DQNAgent(flatten_observation(obs).size, env.action_space.n, DQNConfig())
    metadata = agent.load(str(checkpoint))
    if int(metadata.get("observation_version", 1)) != observation_version:
        raise ValueError(f"Checkpoint observation version mismatch: expected {observation_version}, got {metadata.get('observation_version')}")
    rows=[]
    for seed in seeds:
        obs, _ = env.reset(seed=seed); total_reward=0.0; done=False
        while not done:
            mask=np.asarray(obs["action_mask"], dtype=bool)
            if not mask.any(): break
            action=agent.select_action(obs, explore=False)
            obs,reward,terminated,truncated,_=env.step(action)
            total_reward += float(reward); done=terminated or truncated
        rows.append({"seed":seed,"return":total_reward,**env.get_metrics()})
    return rows

def mean_std(values):
    return {"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)) if len(values)>1 else 0.0}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario",choices=["A","B","C"],default="B")
    p.add_argument("--episodes",type=int,default=100)
    p.add_argument("--train-seeds",type=int,default=3)
    p.add_argument("--eval-seeds",type=int,default=30)
    p.add_argument("--seed-start",type=int,default=0)
    p.add_argument("--output-dir",default="artifacts/observation_v1_v2")
    a=p.parse_args()
    if min(a.episodes,a.train_seeds,a.eval_seeds)<=0: p.error("episodes, train-seeds, and eval-seeds must be positive")
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    evaluation_seeds=list(range(10000,10000+a.eval_seeds))
    for train_seed in range(a.seed_start,a.seed_start+a.train_seeds):
        # Keep each training run's episode seeds disjoint. The model/replay seed
        # and environment seed base are recorded separately for reproducibility.
        environment_seed_start = train_seed * a.episodes
        run_config = {
            "scenario": a.scenario,
            "episodes": a.episodes,
            "model_seed": train_seed,
            "environment_seed_start": environment_seed_start,
            "evaluation_seeds": evaluation_seeds,
        }
        (out/f"seed{train_seed}_config.json").write_text(json.dumps(run_config, indent=2)+"\\n")
        for version in (1,2):
            checkpoint=out/f"v{version}_seed{train_seed}.pt"
            cmd=[sys.executable,"-m","training.train_dqn","--episodes",str(a.episodes),
                 "--seed-start",str(environment_seed_start),"--scenario",a.scenario,
                 "--output",str(checkpoint),"--observation-version",str(version)]
            print("RUN:", " ".join(cmd),flush=True)
            subprocess.run(cmd,check=True)
            eval_rows=evaluate_checkpoint(checkpoint,a.scenario,version,evaluation_seeds)
            eval_path=out/f"v{version}_seed{train_seed}_evaluation.csv"
            with eval_path.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(eval_rows[0]));w.writeheader();w.writerows(eval_rows)
            for row in eval_rows:
                all_rows.append({"observation_version":version,"train_seed":train_seed,**row})
    raw=out/"paired_evaluation.csv"
    with raw.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    summary=[]
    for version in (1,2):
        subset=[r for r in all_rows if r["observation_version"]==version]
        for metric in METRICS:
            summary.append({"observation_version":version,"metric":metric,**mean_std([float(r[metric]) for r in subset])})
    summary_path=out/"summary.json"
    summary_path.write_text(json.dumps({"scenario":a.scenario,"episodes_per_train_seed":a.episodes,
      "train_seeds":list(range(a.seed_start,a.seed_start+a.train_seeds)),"evaluation_seeds":evaluation_seeds,
      "summary":summary},indent=2)+"\n")
    # Paired difference at each evaluation seed after averaging over training seeds.
    paired=[]
    for metric in METRICS:
        v1={s:np.mean([float(r[metric]) for r in all_rows if r["observation_version"]==1 and r["seed"]==s]) for s in evaluation_seeds}
        v2={s:np.mean([float(r[metric]) for r in all_rows if r["observation_version"]==2 and r["seed"]==s]) for s in evaluation_seeds}
        diffs=[v2[s]-v1[s] for s in evaluation_seeds]
        paired.append({"metric":metric,"v2_minus_v1_mean":float(np.mean(diffs)),
                      "v2_minus_v1_std":float(np.std(diffs,ddof=1)) if len(diffs)>1 else 0.0,
                      "evaluation_seed_count":len(diffs)})
    (out/"paired_differences.json").write_text(json.dumps(paired,indent=2)+"\n")
    print(json.dumps({"raw_csv":str(raw),"summary":str(summary_path),"paired_differences":str(out/"paired_differences.json")},indent=2))
if __name__=="__main__": main()
