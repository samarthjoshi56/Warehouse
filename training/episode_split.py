import numpy as np
def episode_holdout_mask(scenarios,seeds,n_holdout=2):
 scenarios=np.asarray(scenarios);seeds=np.asarray(seeds)
 if scenarios.shape!=seeds.shape:raise ValueError("scenario and seed arrays must match")
 if n_holdout<1:raise ValueError("n_holdout must be positive")
 test=np.zeros(len(seeds),dtype=bool)
 for sc in np.unique(scenarios):
  unique=np.unique(seeds[scenarios==sc])
  if len(unique)<=n_holdout:raise ValueError(f"not enough episodes for scenario {sc}")
  test|=(scenarios==sc)&np.isin(seeds,unique[-n_holdout:])
 return test
