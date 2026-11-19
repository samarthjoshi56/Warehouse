# Counterfactual Rollout Horizon Comparison

Date: 2026-11-19

## Results

All means are raw reshuffle counts per 30-item episode.

| Horizon | Seeds/scenario | Improved Greedy overall mean | Counterfactual overall mean | Difference |
|---|---:|---:|---:| 
| Full episode (30-item pilot) | 5 | 6.533 | 3.533 | -3.000 |
| 12 placements | 5 | 6.533 | 6.867 | +0.333 |
| 18 placements | 3 | 6.111 | 4.889 | -1.222 |

## Scenario-level results

### 12-placement horizon, five seeds
| Scenario | Improved Greedy | Counterfactual |
|---|---:|---:|
| A | 9.200 | 8.400 |
| B | 5.600 | 6.200 |
| C | 4.800 | 6.000 |

### 18-placement horizon, three seeds
| Scenario | Improved Greedy | Counterfactual |
|---|---:|---:|
| A | 8.333 | 5.000 |
| B | 6.333 | 5.667 |
| C | 3.667 | 4.000 |

## Runtime
- Five-seed, 30-item full rollout pilot: about 18.86 seconds.
- Five-seed, 30-item 12-placement horizon: about 12.87 seconds.
- Three-seed, 30-item 18-placement horizon: about 10.69 seconds.
- 20-seed full rollout attempt exceeded the 180-second limit and was not completed.

Runtime measurements are specific to this environment.

## Interpretation
The 12-placement horizon did not outperform the heuristic in this pilot. The 18-placement pilot showed a smaller overall improvement but lost in Scenario C and used only three seeds. The full-episode pilot showed a larger improvement but used five seeds.

These results are inconsistent across horizons and remain too small to establish that counterfactual search is better. The 20-seed attempt timed out. The agent is not promoted to deployment.

The code now supports `--rollout-depth`. Set a depth larger than the episode item count to approximate a full-episode rollout.
