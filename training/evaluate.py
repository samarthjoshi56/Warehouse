"""Evaluate DQN and baseline policies on held-out seeds and report statistics."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from agents.dqn import DQNAgent, DQNConfig, flatten_observation
from baselines.strategies import RandomPlacement, FirstFit, GreedyMinHeight, EarliestRetrievalFirst, ImprovedGreedy
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
from evaluation.reporter import StatisticsReporter

def run_policy(env, policy, seed: int, is_dqn: bool = False):
    obs, _ = env.reset(seed=seed); total_reward = 0.0; done = False
    while not done:
        if not np.asarray(obs["action_mask"]).any():
            break
        action = policy.select_action(obs, explore=False) if is_dqn else policy.select_action(obs, obs["action_mask"])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward; done = terminated or truncated
    return {"seed": seed, "return": total_reward, **env.get_metrics()}

def evaluate(checkpoint: str, scenario: str = "B", seeds=range(200, 230), output="artifacts/test_results.csv"):
    seeds = [int(s) for s in seeds]
    if not seeds: raise ValueError("At least one evaluation seed is required")
    env = WarehouseEnv(WarehouseConfig(scenario=scenario))
    obs, _ = env.reset(seed=seeds[0])
    agent = DQNAgent(flatten_observation(obs).size, env.action_space.n, DQNConfig())
    agent.load(checkpoint)
    policies = {"DQN": (agent, True), "Random": (RandomPlacement(17), False),
                "First Fit": (FirstFit(), False), "Greedy Min Height": (GreedyMinHeight(), False),
                "Earliest Retrieval First": (EarliestRetrievalFirst(), False),
                "Improved Greedy": (ImprovedGreedy(), False)}
    rows=[]; grouped={}
    for name, (policy, is_dqn) in policies.items():
        grouped[name]=[]
        for seed in seeds:
            result=run_policy(env, policy, seed, is_dqn)
            grouped[name].append(result); rows.append({"algorithm":name, **result})
    path=Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    reporter=StatisticsReporter(); metric_names=["reshuffles","reshuffles_per_retrieved_item","retrieved","failed","total_retrieval_delay","return"]
    summary=[]
    for metric in metric_names:
        for algorithm, stats in reporter.compare(grouped, metric).items():
            summary.append({"algorithm":algorithm,"metric":metric,**stats})
    summary_path=path.with_name(path.stem+"_summary.csv")
    with summary_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    paired=[]
    dqn_by_seed={int(r["seed"]):r["reshuffles"] for r in grouped["DQN"]}
    comparisons=[name for name in grouped if name!="DQN"]
    for name in comparisons:
        base_by_seed={int(r["seed"]):r["reshuffles"] for r in grouped[name]}
        common=sorted(set(dqn_by_seed)&set(base_by_seed))
        p_value=reporter.paired_wilcoxon([dqn_by_seed[s] for s in common],[base_by_seed[s] for s in common])
        adjusted=min(1.0,p_value*len(comparisons)) if p_value is not None else None
        paired.append({"comparison":"DQN vs "+name,"n":len(common),"wilcoxon_p":p_value,
                       "bonferroni_p":adjusted,"significant_0_05":(adjusted<0.05) if adjusted is not None else None})
    paired_path=path.with_name(path.stem+"_paired_tests.csv")
    with paired_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    print(json.dumps({"raw_results":str(path),"summary":str(summary_path),"paired_tests":str(paired_path),"seeds":seeds},indent=2))
    return rows, summary

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True)
    p.add_argument("--scenario", choices=["A","B","C"], default="B")
    p.add_argument("--output", default="artifacts/test_results.csv")
    a=p.parse_args(); evaluate(a.checkpoint,a.scenario,range(200,230),a.output)
if __name__ == "__main__": main()
