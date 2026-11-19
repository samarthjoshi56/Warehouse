# Counterfactual Score Regression Experiment

## Objective
Predict the counterfactual reshuffle score for each valid placement action, then choose the action with the lowest predicted score. This replaces action classification with outcome-score prediction.

## Data and split
- Dataset: 600 decisions, 20 items per episode, 10 seeds per scenario.
- Input features: flattened simulator observations.
- Target: candidate reshuffle score (first score component).
- Training seeds: 900–907.
- Held-out seeds: 908–909.
- Split unit: seed/episode, not individual decisions.

## Model comparison
The models were trained action-conditionally (one regressor for each action). They were evaluated on the same held-out decisions.

| Model | Action agreement with oracle | Mean cost regret |
|---|---:|---:|
| Random Forest depth 6 | 5.8% | 0.367 |
| Random Forest depth 12 | 5.0% | 0.358 |
| Extra Trees (400 trees, depth 12) | 7.5% | 0.300 |
| HistGradientBoosting | 9.2% | 0.333 |

A reproducible script run with the default Extra Trees configuration (300 trees, depth 8) returned:
- Action agreement: 7.5%
- Mean regret: 0.283
- Selected mean score: 2.408
- Oracle mean score: 2.125

## Interpretation
Action agreement is low. However, the score-regression policy selects actions with an average cost close to the oracle action. This suggests that exact action imitation is a poor metric when multiple actions have similar counterfactual costs.

The experiment is exploratory. It has only two held-out seeds per scenario, so it does not establish generalization. Do not promote this model to the application.

## Reproduce
```bash
python training/train_counterfactual_score_regression.py --model extra
python training/train_counterfactual_score_regression.py --model rf
python training/train_counterfactual_score_regression.py --model histgb
```

The script saves an NPZ checkpoint containing the per-action regressors and metrics.
