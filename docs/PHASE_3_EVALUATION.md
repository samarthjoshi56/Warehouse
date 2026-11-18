# Phase 3 — DQN Training and Held-Out Evaluation

**Status:** Initial DQN implementation trained and evaluated. The current DQN is not the best-performing policy.

## Implementation

- Masked Double DQN with a two-layer MLP.
- Experience replay and target network.
- Epsilon-greedy exploration.
- Action masks during exploration and greedy action selection.
- Checkpointing and per-episode CSV output.
- Evaluation against all five baseline policies.

## Reproducibility

- Scenario: B (noisy expected retrieval times).
- Training seeds: 0–49 (50 episodes).
- Test seeds: 200–229 (30 held-out episodes).
- Warehouse defaults: 10 stacks, height 5, temporary storage 10, 100 items.
- Primary metric: total reshuffles per episode. Lower is better.
- Checkpoint: `artifacts/dqn_warehouse.pt` (generated locally; excluded from Git).

## Test-set results

| Policy | Mean reshuffles | Median | Std. dev. | 95% CI |
|---|---:|---:|---:|---:|
| DQN | 12.27 | 13.0 | 1.60 | [11.70, 12.84] |
| Random | 10.50 | 10.5 | 1.50 | [9.96, 11.04] |
| First Fit | 13.57 | 13.0 | 1.74 | [12.95, 14.19] |
| Greedy Min Height | 8.37 | 8.5 | 2.24 | [7.57, 9.17] |
| Earliest Retrieval First | 9.23 | 9.0 | 2.86 | [8.21, 10.26] |
| Improved Greedy | 7.67 | 7.0 | 2.54 | [6.76, 8.57] |

All policies retrieved all 100 items in every evaluated episode.

## Interpretation

The DQN policy did not outperform the heuristic baselines. Its mean reshuffle count was higher than Random, Greedy Min Height, Earliest Retrieval First, and Improved Greedy. Paired Wilcoxon tests on the 30 common test seeds found worse DQN results than Improved Greedy (p = 3.40e-6), Greedy Min Height (p = 2.34e-6), Random (p = 1.04e-4), and Earliest Retrieval First (p = 2.11e-5). These comparisons remain significant under a Bonferroni threshold of 0.0125.

Do not describe this model as better than the baselines. Treat it as a working training pipeline and a weak initial policy. The heuristic baselines remain the current performance reference.

## Known limitations

1. Feed-forward DQN does not model the full event history in Scenario B, which is partially observable.
2. The finite lookahead buffer in Scenario C does not expose the complete scheduler queue.
3. Placement shaping is heuristic and is not policy-invariant.
4. This is one fixed hyperparameter run. No test-set tuning was performed.
5. The simulator still needs broader coverage against the complete Phase 1 test specification.

## Next research steps

Use validation seeds 100–119 for future hyperparameter changes. Consider recurrent policies or history stacking for Scenarios A and B. Keep seeds 200–229 locked after this report; use a new held-out test set for a future final comparison.
