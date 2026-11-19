# Guarded Hybrid Policy Experiment

Date: 2026-11-19

## Objective
Test whether DQN can safely override Improved Greedy when its Q-value margin exceeds a threshold.

## Method
- Policies: Improved Greedy, standalone DQN, and guarded hybrid.
- Scenarios: A, B, C.
- Seeds: 500–511 in each scenario.
- Items per episode: 100.
- Hybrid Q-margin thresholds: 0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10.
- Total: 396 episodes.
- Action masking was applied to invalid stacks.

## Overall mean reshuffles

| Policy | Mean reshuffles | Mean override rate |
|---|---:|---:|
| Improved Greedy | 9.111 | — |
| DQN | 12.361 | — |
| Hybrid margin 0 | 13.167 | 98.8% |
| Hybrid margin 0.05 | 12.917 | 91.2% |
| Hybrid margin 0.1 | 12.778 | 83.7% |
| Hybrid margin 0.25 | 11.750 | 64.0% |
| Hybrid margin 0.5 | 10.972 | 36.9% |
| Hybrid margin 1 | 10.333 | 7.4% |
| Hybrid margin 2 | 9.639 | 2.2% |
| Hybrid margin 5 | 9.111 | 0.0% |
| Hybrid margin 10 | 9.111 | 0.0% |

## Scenario-level means

| Scenario | Improved Greedy | DQN | Hybrid 0.5 | Hybrid 2 | Hybrid 5 |
|---|---:|---:|---:|---:|---:|
| A | 10.667 | 12.833 | 11.917 | 10.667 | 10.667 |
| B | 9.167 | 12.000 | 10.667 | 9.833 | 9.167 |
| C | 7.500 | 12.250 | 10.333 | 8.417 | 7.500 |

## Interpretation
DQN overrides degraded performance at low and medium thresholds. At thresholds 5 and 10, the hybrid did not override during these test episodes and matched Improved Greedy. Q-margin is not calibrated as a probability and did not prove useful as an override criterion.

The hybrid is not promoted. Improved Greedy remains the reference policy.

## Implementation
- `agents/hybrid.py`
- `training/evaluate_hybrid_thresholds.py`
- `tests/test_hybrid.py`

## Test status
29 automated tests passed after the action-mask correction.
