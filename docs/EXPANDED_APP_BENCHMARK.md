# Expanded Application Benchmark (30 seeds)

Date: 2026-11-19

## Method
- Policies: Random, First Fit, Greedy (Min Height), Earliest Retrieval First, Improved Greedy, Trained DQN.
- Scenarios: A, B, C.
- Seeds: 0–29 for each policy/scenario.
- Items per episode: 100.
- Total: 540 episodes and 54,000 items.
- Metrics: reshuffles. Failure, unplaced-item, and invalid-action counts were also checked.

## Mean reshuffles per episode

| Policy | Scenario A | Scenario B | Scenario C | Overall |
|---|---:|---:|---:|---:|
| Random | 12.33 | 11.90 | 11.27 | 11.83 |
| First Fit | 14.50 | 13.67 | 13.63 | 13.93 |
| Greedy (Min Height) | 10.43 | 10.10 | 9.97 | 10.17 |
| Earliest Retrieval First | 14.50 | 9.33 | 8.57 | 10.80 |
| Improved Greedy | 10.43 | 8.63 | 6.40 | 8.49 |
| Trained DQN | 13.07 | 13.00 | 13.07 | 13.04 |

## Integrity
- Failed items: 0.
- Unplaced items: 0.
- Invalid actions: 0.

## Interpretation
Improved Greedy has the lowest overall mean. The trained DQN did not outperform the strongest heuristic. These results are diagnostic and do not establish broad generalization.
