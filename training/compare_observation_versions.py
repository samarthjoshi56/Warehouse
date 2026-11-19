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

def environment_seed_start_for_run(train_seed: int, episodes: int) -> int:
    """Allocate non-overlapping environment seed blocks to training runs."""
    return int(train_seed) * int(episodes)

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
        environment_seed_start = environment_seed_start_for_run(train_seed, a.episodes)
        run_config = {
            "scenario": a.scenario,
            "episodes": a.episodes,
            "model_seed": environment_seed_start,
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
    # Pair V1 and V2 within each training seed and evaluation seed.
    # Report evaluation uncertainty per training run, then summarize across runs.
    paired_by_train_seed = []
    for train_seed in range(a.seed_start, a.seed_start+a.train_seeds):
        for metric in METRICS:
            diffs = []
            for eval_seed in evaluation_seeds:
                v1 = [float(r[metric]) for r in all_rows if r["observation_version"] == 1 and r["train_seed"] == train_seed and r["seed"] == eval_seed]
                v2 = [float(r[metric]) for r in all_rows if r["observation_version"] == 2 and r["train_seed"] == train_seed and r["seed"] == eval_seed]
                if len(v1) != 1 or len(v2) != 1:
                    raise ValueError(f"Missing or duplicate paired evaluation: train_seed={train_seed}, eval_seed={eval_seed}, metric={metric}")
                diffs.append(v2[0] - v1[0])
            rng = np.random.default_rng(2026 + train_seed)
            boot = np.mean(rng.choice(np.asarray(diffs), size=(10000, len(diffs)), replace=True), axis=1)
            paired_by_train_seed.append({"metric":metric,"train_seed":train_seed,
                "mean_v2_minus_v1":float(np.mean(diffs)),"evaluation_seed_count":len(diffs),
                "bootstrap_95_ci":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))]})
    overall_paired = []
    for metric in METRICS:
        effects = [r["mean_v2_minus_v1"] for r in paired_by_train_seed if r["metric"] == metric]
        overall_paired.append({"metric":metric,"mean_train_seed_effect":float(np.mean(effects)),
            "std_across_train_seeds":float(np.std(effects,ddof=1)) if len(effects)>1 else 0.0,
            "train_seed_count":len(effects)})
    (out/"paired_differences.json").write_text(json.dumps({"per_train_seed":paired_by_train_seed,
        "overall_across_train_seeds":overall_paired},indent=2)+"\n")
    print(json.dumps({"raw_csv":str(raw),"summary":str(summary_path),"paired_differences":str(out/"paired_differences.json")},indent=2))
if __name__=="__main__": main()
