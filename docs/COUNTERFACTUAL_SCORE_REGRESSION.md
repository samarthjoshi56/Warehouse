# Counterfactual Score Regression Experiment

## Objective
Predict the counterfactual reshuffle score for each valid placement action, then choose the action with the lowest predicted score. This replaces action classification with outcome-score prediction.

## Dataset 1: 600 decisions
- 20 items per episode, 10 seeds per scenario.
- Training seeds: 900–907.
- Held-out seeds: 908–909.
- Split unit: seed/episode, not individual decisions.

| Model | Action agreement with oracle | Mean cost regret |
|---|---:|---:|
| Random Forest depth 6 | 5.8% | 0.367 |
| Random Forest depth 12 | 5.0% | 0.358 |
| Extra Trees (400 trees, depth 12) | 7.5% | 0.300 |
| HistGradientBoosting | 9.2% | 0.333 |

Default Extra Trees run (300 trees, depth 8): agreement 7.5%, regret 0.283, selected mean score 2.408, oracle mean score 2.125.

## Dataset 2: 900 decisions
- 45 episodes: 15 seeds per scenario, 20 items per episode.
- Training seeds: 930–942.
- Held-out seeds: 943–944.
- Split unit: episode seed.

| Model | Action agreement with oracle | Mean cost regret |
|---|---:|---:|
| Random Forest | 12.5% | 0.367 |
| Extra Trees | 15.0% | **0.325** |
| HistGradientBoosting | **19.2%** | 0.358 |

Extra Trees has the lowest mean regret in this test. HistGradientBoosting matches the oracle action more often, but has higher regret. This shows why action agreement alone is not a sufficient metric.

## Interpretation
Action agreement remains modest. However, the models select actions with average costs reasonably close to the oracle actions. Exact action imitation can be misleading when several actions have similar costs.

Results are exploratory. Both held-out sets contain only two seeds per scenario. Do not promote these models to the demo. A larger, independently held-out evaluation is required.

## Reproduce
```bash
python training/train_counterfactual_score_regression.py --model extra
python training/train_counterfactual_score_regression.py --model rf
python training/train_counterfactual_score_regression.py --model histgb
```
The script saves an NPZ checkpoint containing the per-action regressors and metrics.
