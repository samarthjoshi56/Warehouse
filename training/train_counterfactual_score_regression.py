"""Train action-conditioned counterfactual score regressors with episode-level splits.
Each action model predicts the counterfactual reshuffle score for that action.
The policy chooses the valid action with the lowest predicted score. Seeds, not
individual decisions, define the held-out split to reduce episode leakage.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='artifacts/counterfactual_labels_10seed.npz')
    p.add_argument('--output', default='artifacts/counterfactual_score_model.npz')
    p.add_argument('--trees', type=int, default=300)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--min-leaf', type=int, default=3)
    p.add_argument('--holdout-seeds', type=int, default=2)
    p.add_argument('--model', choices=['rf', 'extra', 'histgb'], default='extra')
    args = p.parse_args()
    d = np.load(args.data)
    X = d['states'].astype(np.float32)
    masks = d['action_masks'].astype(bool)
    scores = d['candidate_scores'][:, :, 0].astype(np.float32)
    seeds = d['seeds']
    unique_seeds = np.unique(seeds)
    holdout = set(unique_seeds[-args.holdout_seeds:].tolist())
    test = np.isin(seeds, list(holdout))
    train = ~test
    models = {}
    pred = np.full_like(scores, np.nan, dtype=np.float32)
    for action in range(masks.shape[1]):
        eligible = train & masks[:, action] & np.isfinite(scores[:, action])
        if eligible.sum() < 10:
            continue
        if args.model == 'rf':
            model = RandomForestRegressor(n_estimators=args.trees, max_depth=args.depth,
                min_samples_leaf=args.min_leaf, random_state=action, n_jobs=-1)
        elif args.model == 'extra':
            model = ExtraTreesRegressor(n_estimators=args.trees, max_depth=args.depth,
                min_samples_leaf=args.min_leaf, random_state=action, n_jobs=-1)
        else:
            model = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=20,
                min_samples_leaf=5, random_state=action)
        model.fit(X[eligible], scores[eligible, action])
        models[action] = model
        pred[test, action] = model.predict(X[test])
    valid_test = test & masks.any(axis=1)
    valid_pred = np.where(masks[valid_test], pred[valid_test], np.inf)
    chosen = np.argmin(valid_pred, axis=1)
    oracle = np.where(masks[valid_test], scores[valid_test], np.inf).argmin(axis=1)
    chosen_scores = scores[valid_test, chosen]
    oracle_scores = scores[valid_test, oracle]
    scenario_ids = d['scenario_ids'][valid_test]
    metrics = {
        'model': args.model, 'samples': int(len(X)),
        'train_decisions': int(train.sum()), 'test_decisions': int(test.sum()),
        'train_seeds': sorted(set(seeds[train].tolist())),
        'test_seeds': sorted(holdout),
        'test_action_agreement': float(np.mean(chosen == oracle)),
        'mean_regret_reshuffles': float(np.mean(chosen_scores - oracle_scores)),
        'mean_selected_score': float(np.mean(chosen_scores)),
        'mean_oracle_score': float(np.mean(oracle_scores)),
        'per_scenario': {
            str(int(s)): {
                'action_agreement': float(np.mean(chosen[scenario_ids == s] == oracle[scenario_ids == s])),
                'regret': float(np.mean(chosen_scores[scenario_ids == s] - oracle_scores[scenario_ids == s]))
            } for s in np.unique(scenario_ids)
        },
        'action_models': sorted(models.keys())
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, models=np.array([models], dtype=object),
        action_dim=masks.shape[1], input_dim=X.shape[1], metrics=np.array([metrics], dtype=object))
    print(json.dumps(metrics, indent=2))
    print('saved:', out)

if __name__ == '__main__':
    main()
