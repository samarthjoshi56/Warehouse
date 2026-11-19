# Guarded Hybrid Policy Experiment — Extended Validation

Date: 2026-11-19

## Objective
Test whether the DQN can safely override Improved Greedy when its Q-value margin exceeds a threshold.

## Method
- Policies: Improved Greedy, standalone DQN, guarded hybrid.
- Scenarios: A, B, C.
- Seeds: 500–529 in each scenario (paired seeds).
- Items per episode: 100.
- Thresholds: 0.5, 2, and 5.
- Total: 450 episodes, 45,000 items.

## Overall mean reshuffles

| Policy | Mean reshuffles | Mean override rate |
|---|---:|---:|
| Improved Greedy | 7.833 | — |
| Hybrid margin 5 | 7.833 | 0% |
| Hybrid margin 2 | 8.422 | 2.17% |
| Hybrid margin 0.5 | 10.178 | 36.41% |
| DQN | 11.700 | — |

## Scenario means

| Scenario | Improved Greedy | DQN | Hybrid 0.5 | Hybrid 2 | Hybrid 5 |
|---|---:|---:|---:|---:|---:|
| A | 9.533 | 11.867 | 10.267 | 9.533 | 9.533 |
| B | 7.867 | 11.533 | 10.400 | 8.700 | 7.867 |
| C | 6.100 | 11.700 | 9.867 | 7.033 | 6.100 |

## Paired-seed comparison
- Hybrid margin 0.5 vs Improved Greedy: +2.344 reshuffles on average; 12 better, 13 ties, 65 worse episodes.
- Hybrid margin 2 vs Improved Greedy: +0.589 reshuffles on average; 8 better, 47 ties, 35 worse episodes.
- Hybrid margin 5 vs Improved Greedy: identical in all 90 paired episodes because it made no overrides.
- DQN vs Improved Greedy: +3.867 reshuffles on average; 6 better, 5 ties, 79 worse episodes.

These are exploratory paired comparisons. The seed sample is limited and was used to guide experiments, not establish formal generalization.

## Integrity
All 45,000 items were retrieved. Failed, unplaced, and invalid-action counts were zero.

## Decision
Do not promote the hybrid or DQN. Improved Greedy remains the best tested policy. Q-value margin did not provide a useful override rule in this experiment.

## Files
- `agents/hybrid.py`
- `training/evaluate_hybrid_thresholds.py`
- `tests/test_hybrid.py`
- `artifacts/hybrid_threshold_sweep.csv` (episode-level data)
- `artifacts/hybrid_threshold_summary.csv` (scenario-level summary)
