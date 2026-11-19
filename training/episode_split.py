"""Episode-level train/test split utilities for warehouse datasets."""
from __future__ import annotations
import numpy as np


def episode_holdout_mask(scenario_ids, seeds, holdout_per_scenario=2):
    """Hold out the same number of distinct seeds independently per scenario.

    Returns a boolean test mask. Episodes are identified by (scenario, seed), so
    decisions from one episode cannot appear on both sides of the split.
    """
    scenario_ids=np.asarray(scenario_ids)
    seeds=np.asarray(seeds)
    if scenario_ids.shape != seeds.shape:
        raise ValueError("scenario_ids and seeds must have the same shape")
    if holdout_per_scenario < 1:
        raise ValueError("holdout_per_scenario must be >= 1")
    test=np.zeros(len(seeds),dtype=bool)
    for scenario in np.unique(scenario_ids):
        episode_seeds=np.unique(seeds[scenario_ids==scenario])
        if len(episode_seeds) <= holdout_per_scenario:
            raise ValueError(f"scenario {scenario} has too few episodes for holdout")
        holdout=set(episode_seeds[-holdout_per_scenario:].tolist())
        test |= (scenario_ids==scenario) & np.isin(seeds,list(holdout))
    return test
