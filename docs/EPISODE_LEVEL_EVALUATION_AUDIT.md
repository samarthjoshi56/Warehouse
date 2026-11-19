# Episode-Level Evaluation Audit

## Why this change
Earlier counterfactual imitation experiments split individual decisions into train and validation sets. Since decisions from the same simulated episode are correlated, this can leak episode-specific information across the split and inflate validation results.

## Correct split
The new utility `training/episode_split.py` holds out complete episodes independently within each scenario. An episode is identified by the pair `(scenario_id, seed)`. The last N unique seeds per scenario are held out, where N is configurable.

## Updated training/evaluation scripts
- `training/train_counterfactual_rf.py`
- `training/train_counterfactual_bc.py`
- `training/train_counterfactual_score_regression.py`

These scripts now use complete episode keys for held-out validation. The split tests check the per-scenario holdout and reject insufficient episode counts.

## Important limitation
Earlier decision-level validation results must not be treated as clean estimates of generalisation. They are exploratory. Re-run the experiments with the corrected split before using their validation metrics in reports or decisions.

## Reproduce
```bash
python -m pytest -q
python training/train_counterfactual_rf.py --holdout-seeds 2
python training/train_counterfactual_bc.py --holdout-seeds 2
python training/train_counterfactual_score_regression.py --holdout-seeds 2 --model extra
```
