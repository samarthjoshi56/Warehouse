# Counterfactual Placement Rollout Experiment

Date: 2026-11-19

## Method
At each placement decision, the agent deep-copies the live environment for every valid stack action, applies that action, then completes the simulated episode using Improved Greedy. It selects the action with the lowest final reshuffle count, breaking ties by failed items, unplaced items, and action ID.

This is a model-based rollout policy. It does not train a neural network and it uses the simulator's internal state through environment copying. It is therefore a privileged-information research baseline, not a deployable policy under the observation-only interface.

## Pilot
- Scenarios: A, B, C
- Seeds: 700, 701, 702
- Items per episode: 30
- Total episodes: 18 (baseline and counterfactual for each scenario/seed)
- Total processed items: 540

## Results

| Scenario | Improved Greedy mean | Counterfactual mean | Difference |
|---|---:|---:|---:|
| A | 8.333 | 4.333 | -4.000 |
| B | 6.333 | 3.333 | -3.000 |
| C | 3.667 | 1.333 | -2.333 |
| Overall | 6.111 | 3.000 | -3.111 |

## Integrity
- All 540 items retrieved.
- Failed items: 0.
- Unplaced items: 0.
- Invalid actions: 0.

## Interpretation and limitations
The pilot shows lower reshuffle counts for counterfactual rollout on these 3 seeds. The sample is too small for a reliable conclusion. The policy is computationally expensive because it runs one full rollout for each valid action at every decision. It also uses cloned simulator internals, so it is not a fair observation-only deployable policy. Treat it as an oracle-style baseline and use it to study decision quality and generate training targets.

Next steps: increase seed count, measure wall-clock cost, compare rollout policies, and use counterfactual action scores as targets for a learned model. Do not promote this agent to the demo as a realistic deployable policy.
