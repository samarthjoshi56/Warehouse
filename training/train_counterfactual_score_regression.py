"""Counterfactual score regression with scenario-stratified episode holdout."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from training.episode_split import episode_holdout_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/counterfactual_labels_10seed.npz")
    p.add_argument("--output", default="artifacts/counterfactual_score_model.npz")
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--min-leaf", type=int, default=3)
    p.add_argument("--holdout-seeds", type=int, default=2)
    p.add_argument("--model", choices=["rf", "extra", "histgb"], default="extra")
    a = p.parse_args()

    d = np.load(a.data)
    X = d["states"].astype(np.float32)
    mask = d["action_masks"].astype(bool)
    scores = d["candidate_scores"][:, :, 0].astype(np.float32)
    scenarios, seeds = d["scenario_ids"], d["seeds"]
    test = episode_holdout_mask(scenarios, seeds, a.holdout_seeds)
    train = ~test

    models = {}
    pred = np.full_like(scores, np.nan)
    for action in range(mask.shape[1]):
        eligible = train & mask[:, action] & np.isfinite(scores[:, action])
        if eligible.sum() < 10:
            continue
        if a.model == "rf":
            model = RandomForestRegressor(
                n_estimators=a.trees, max_depth=a.depth,
                min_samples_leaf=a.min_leaf, random_state=action, n_jobs=-1,
            )
        elif a.model == "extra":
            model = ExtraTreesRegressor(
                n_estimators=a.trees, max_depth=a.depth,
                min_samples_leaf=a.min_leaf, random_state=action, n_jobs=-1,
            )
        else:
            model = HistGradientBoostingRegressor(
                max_iter=150, max_leaf_nodes=20, min_samples_leaf=5, random_state=action,
            )
        model.fit(X[eligible], scores[eligible, action])
        models[action] = model
        pred[test, action] = model.predict(X[test])

    # A decision is evaluable only if at least one legal action has a prediction.
    valid_test = test & mask.any(axis=1)
    masked_predictions = np.where(mask[valid_test], pred[valid_test], np.inf)
    masked_predictions[~np.isfinite(masked_predictions)] = np.inf
    usable = np.isfinite(masked_predictions).any(axis=1)
    if not usable.any():
        raise RuntimeError("No held-out decisions have predictions for legal actions")
    masked_predictions = masked_predictions[usable]
    decision_mask = mask[valid_test][usable]
    actual_scores = scores[valid_test][usable]
    scenario_test = scenarios[valid_test][usable]
    seed_test = seeds[valid_test][usable]

    chosen = masked_predictions.argmin(axis=1)
    oracle_scores = np.where(decision_mask, actual_scores, np.inf)
    oracle = oracle_scores.argmin(axis=1)
    selected = actual_scores[np.arange(len(chosen)), chosen]
    best = actual_scores[np.arange(len(chosen)), oracle]
    metrics = {
        "model": a.model,
        "samples": int(len(X)),
        "train_decisions": int(train.sum()),
        "test_decisions": int(test.sum()),
        "evaluated_decisions": int(len(chosen)),
        "train_episodes": len(set(zip(scenarios[train].tolist(), seeds[train].tolist()))),
        "test_episodes": len(set(zip(scenarios[test].tolist(), seeds[test].tolist()))),
        "test_episode_keys": sorted(set((int(x), int(y)) for x, y in zip(scenarios[test], seeds[test]))),
        "action_agreement": float(np.mean(chosen == oracle)),
        "mean_regret": float(np.mean(selected - best)),
        "mean_selected_score": float(np.mean(selected)),
        "mean_oracle_score": float(np.mean(best)),
        "per_scenario": {
            str(int(sc)): {
                "agreement": float(np.mean(chosen[scenario_test == sc] == oracle[scenario_test == sc])),
                "regret": float(np.mean(selected[scenario_test == sc] - best[scenario_test == sc])),
                "decisions": int(np.sum(scenario_test == sc)),
            }
            for sc in np.unique(scenario_test)
        },
    }
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, models=np.array([models], dtype=object),
        action_dim=mask.shape[1], input_dim=X.shape[1],
        metrics=np.array([metrics], dtype=object),
    )
    print(json.dumps(metrics, indent=2))
    print("saved:", output)


if __name__ == "__main__":
    main()
