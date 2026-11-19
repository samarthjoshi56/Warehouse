# Counterfactual Label Learning: Neural and Tree Baselines

Date: 2026-11-19

## Label collection
- Full-rollout counterfactual labels: 600 decisions.
- 10 seeds per scenario (A/B/C), seeds 900–909.
- 20 items per episode; 30 episodes total.
- Counterfactual label selected by simulating each valid action through the episode with Improved Greedy continuation.
- Collection time: 16.77 seconds in this runtime.

## Neural behavior clone
- Random 80/20 decision-level split: 480 training, 120 validation samples.
- 100 epochs; 128-hidden-unit network.
- Validation accuracy: 30.0%.
- Held-out episode seeds: 910–913, 20 items each, all scenarios.

| Scenario | Improved Greedy mean | Neural clone mean |
|---|---:|---:|
| A | 5.00 | 8.50 |
| B | 4.00 | 7.50 |
| C | 3.50 | 8.00 |
| Overall | 4.17 | 8.00 |

## Random forest baseline
- 300 trees, maximum depth 6, minimum leaf size 4, balanced class weights.
- Decision-level validation accuracy: 32.5%.
- Same held-out episode seeds as neural clone.

| Scenario | Improved Greedy mean | Random forest mean |
|---|---:|---:|
| A | 5.00 | 8.75 |
| B | 4.00 | 7.25 |
| C | 3.50 | 7.00 |
| Overall | 4.17 | 7.67 |

## Integrity
- All 240 items per policy were retrieved (720 total across Improved Greedy, neural clone, and random forest).
- Failed, unplaced, and invalid actions were zero.

## Conclusion
Both learned models failed to generalize well from this small counterfactual dataset. The random forest slightly improved classification accuracy but did not improve the operational metric. This is consistent with the earlier hybrid and behavior-cloning experiments: action classification accuracy alone is not enough. The counterfactual labels are noisy and depend on the rollout continuation policy. Decision-level random validation also has correlated samples from the same episode, so it is optimistic as a generalization estimate. Use episode-level splits and more diverse training seeds in the next iteration. Neither learned checkpoint is suitable for deployment.
