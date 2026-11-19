# Blocker-Aware Placement Experiment

## Purpose
Test a simple, explainable heuristic before adding another learned policy.

## Policy
`BlockerAwarePlacement` scores each valid stack by estimated retrieval blockers, stack height, and a small temporary-storage pressure term. If the incoming retrieval time is unknown, it falls back to stack-height scoring. The policy is research-only and is not enabled in the application.

## Evaluation design
Paired seeds compare Blocker-Aware with Improved Greedy. The same scenario and seed are used for both policies. Reshuffles are the primary metric. These runs are exploratory, not formal statistical validation.

## Results

### 20 items, 30 seeds per scenario (seeds 1000–1029)
| Scenario | Improved Greedy | Blocker-Aware | Paired mean delta |
|---|---:|---:|---:|
| A | 5.867 | 5.867 | 0.000 |
| B | 4.567 | 4.333 | -0.233 |
| C | 2.900 | 2.800 | -0.100 |

Scenario B: 9 wins, 19 ties, 2 losses. Scenario C: 5 wins, 23 ties, 2 losses.

### 40 items, 30 seeds per scenario (seeds 2000–2029)
| Scenario | Improved Greedy | Blocker-Aware | Paired mean delta |
|---|---:|---:|---:|
| A | 9.500 | 9.500 | 0.000 |
| B | 7.567 | 7.600 | +0.033 |
| C | 5.800 | 5.533 | -0.267 |

Scenario B: 9 wins, 13 ties, 8 losses. Scenario C: 11 wins, 13 ties, 6 losses.

### 100 items, 15 seeds per scenario (seeds 3000–3014)
| Scenario | Improved Greedy | Blocker-Aware | Paired mean delta |
|---|---:|---:|---:|
| A | 10.467 | 10.467 | 0.000 |
| B | 9.533 | 9.467 | -0.067 |
| C | 6.267 | 6.400 | +0.133 |

Scenario B: 7 wins, 5 ties, 3 losses. Scenario C: 3 wins, 7 ties, 5 losses.

## Conclusion
The heuristic does not improve Scenario A because retrieval times are unknown and the rule reduces to height preference. It gives small, inconsistent changes in Scenarios B and C. Results do not support replacing Improved Greedy. Keep the agent research-only.

## Reproduction
```bash
PYTHONPATH=. python training/evaluate_blocker_aware.py --seeds 30 --items 20
PYTHONPATH=. python training/evaluate_blocker_aware.py --seeds 30 --items 40 --start-seed 2000 --output artifacts/blocker_aware_40items.csv
PYTHONPATH=. python training/evaluate_blocker_aware.py --seeds 15 --items 100 --start-seed 3000 --output artifacts/blocker_aware_100items.csv
```
