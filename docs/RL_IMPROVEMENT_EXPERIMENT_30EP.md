# RL Improvement Experiment — 30-Episode Run

Date: 2026-11-19

## Summary

A fresh DQN was trained for 30 episodes in Scenario B, using seeds 3000–3029. It was evaluated on held-out seeds 200–229 in all three scenarios.

The experiment checkpoint is **not** promoted to the deployment checkpoint. It shows a small improvement in Scenario B, but still does not beat the heuristic policies.

## Scenario B: same-seed comparison

| Policy / checkpoint | Mean reshuffles (30 seeds) | Standard deviation |
|---|---:|---:|
| Original DQN | 12.267 | 1.691 |
| New 30-episode DQN | 11.633 | 1.691 |
| Random | 10.500 | 1.503 |
| First Fit | 13.567 | 1.736 |
| Greedy Min Height | 8.367 | 2.236 |
| Earliest Retrieval First | 9.233 | 2.861 |
| Improved Greedy | 7.667 | 2.537 |

The experimental DQN reduced reshuffles by 0.633 per 100 items versus the original checkpoint, on the same held-out seeds. A one-sided paired Wilcoxon test gives p = 0.0210 for improvement. This is a limited experiment, not proof of general improvement.

## New DQN across scenarios

| Scenario | Mean DQN reshuffles | Random | First Fit | Greedy Min Height | Earliest Retrieval First | Improved Greedy |
|---|---:|---:|---:|---:|---:|
| A | 14.333 | 10.500 | 13.467 | 8.733 | 13.467 | 8.733 |
| B | 11.633 | 10.500 | 13.567 | 8.367 | 9.233 | 7.667 |
| C | 11.467 | 10.033 | 12.633 | 8.233 | 7.600 | 5.400 |

Scenario A and C results show weak generalization. The trained DQN performs worse than the stronger heuristic in all three scenarios.

## App smoke-test matrix

Tested every policy in scenarios A, B, and C with 40 items and seed 123: 18 combinations.

- 18/18 runs completed.
- 40/40 items retrieved in every run.
- 0 failed items.
- 0 unplaced items.
- 0 invalid actions.

## Checkpoint

- Experimental file: `artifacts/dqn_experiment_30ep.pt`
- Training seeds: 3000–3029
- Training episodes: 30
- Evaluation seeds: 200–229
- Scenario B evaluation is held-out from training seeds.

## Diagnosis

The current DQN policy is not competitive. The next experiments should focus on representation and learning setup, not only episode count:

1. Establish a transparent reward breakdown and inspect whether reshuffle penalties dominate or are delayed.
2. Add state feature normalization / simplify the input representation.
3. Compare a supervised imitation baseline trained on Improved Greedy decisions before RL fine-tuning.
4. Train across scenarios instead of only Scenario B.
5. Use longer runs only after an experiment demonstrates a useful trend.

No experimental checkpoint was substituted for the deployment checkpoint.
