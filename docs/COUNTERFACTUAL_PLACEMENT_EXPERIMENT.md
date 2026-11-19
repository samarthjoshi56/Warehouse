# Counterfactual Placement Rollout Experiment

Date: 2026-11-19

## Method
At each placement decision, the agent deep-copies the live environment for every valid stack action, applies that action, then completes the simulated episode using Improved Greedy. It selects the action with the lowest final reshuffle count, breaking ties by failed items, unplaced items, and action ID.

This is a model-based rollout policy. It does not train a neural network and it uses simulator internals through environment copying. It is therefore a privileged-information research baseline, not a deployable policy under the observation-only interface.

## Pilot
- Scenarios: A, B, C
- Seeds: 700–704
- Items per episode: 30
- Total episodes: 30 (baseline and counterfactual for each scenario/seed)
- Total processed items: 900
- Wall-clock time: 18.86 seconds in this runtime for the benchmark
- One 60-item Scenario C counterfactual episode: 6.76 seconds (single measurement; not a stable performance estimate)

## Results
Values are mean reshuffle counts per 30-item episode (not normalized per 100 items).

| Scenario | Improved Greedy mean | Counterfactual mean | Difference |
|---|---:|---:|---:|
| A | 9.200 | 5.200 | -4.000 |
| B | 5.600 | 3.000 | -2.600 |
| C | 4.800 | 2.400 | -2.400 |
| Overall | 6.533 | 3.533 | -3.000 |

Paired episode differences (Counterfactual minus Improved Greedy): 14 counterfactual wins, 1 tie, and 0 losses across 15 paired scenario-seed cases. Mean difference is -3.000 reshuffles per 30-item episode.

## Integrity
- All 900 items retrieved.
- Failed items: 0.
- Unplaced items: 0.
- Invalid actions: 0.

## Interpretation and limitations
The counterfactual rollout pilot improves over Improved Greedy on these 15 paired scenario-seed cases. This is a small pilot. It is not sufficient to establish statistical generalization. It is computationally expensive and uses cloned simulator internals, so it is not a fair observation-only deployable policy. Treat it as an oracle-style baseline and use it to generate training targets.

Next steps: increase seed count, measure runtime across episode lengths, compare rollout policies, and use counterfactual action scores as targets for a learned model. Do not promote this agent to the demo as a realistic deployable policy.
