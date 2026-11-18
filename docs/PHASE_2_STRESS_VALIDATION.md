# Phase 2 — Multi-Scenario Stress Validation

**Purpose:** Check that the simulator and heuristic policies complete seeded episodes across the three information scenarios. This is a smoke/stress check, not the final statistical benchmark.

## Protocol

- 3 information scenarios: A, B, C.
- 5 heuristic policies: Random, First Fit, Greedy Min Height, Earliest Retrieval First, Improved Greedy.
- 30 seeds per policy and scenario (seeds 0–29).
- 40 items per episode.
- 450 total episodes.
- 18,000 total items.

## Results

| Scenario | Policy | Mean reshuffles | Items retrieved | Failed | Unplaced |
|---|---|---:|---:|---:|---:|
| A | Random | 12.33 | 1,200 | 0 | 0 |
| A | First Fit | 14.47 | 1,200 | 0 | 0 |
| A | Greedy Min Height | 10.43 | 1,200 | 0 | 0 |
| A | Earliest Retrieval First | 14.47 | 1,200 | 0 | 0 |
| A | Improved Greedy | 10.43 | 1,200 | 0 | 0 |
| B | Random | 11.90 | 1,200 | 0 | 0 |
| B | First Fit | 13.67 | 1,200 | 0 | 0 |
| B | Greedy Min Height | 10.10 | 1,200 | 0 | 0 |
| B | Earliest Retrieval First | 9.30 | 1,200 | 0 | 0 |
| B | Improved Greedy | 8.63 | 1,200 | 0 | 0 |
| C | Random | 11.27 | 1,200 | 0 | 0 |
| C | First Fit | 13.63 | 1,200 | 0 | 0 |
| C | Greedy Min Height | 9.97 | 1,200 | 0 | 0 |
| C | Earliest Retrieval First | 8.53 | 1,200 | 0 | 0 |
| C | Improved Greedy | 6.40 | 1,200 | 0 | 0 |

All 450 episodes completed all 40 retrievals. No item was failed or left unplaced. The run also checked item conservation after every environment step.

## Interpretation

- Improved Greedy had the lowest mean reshuffle count in Scenarios B and C.
- In Scenario A, Greedy Min Height and Improved Greedy had equal means.
- First Fit was the weakest heuristic in these runs.
- These values are descriptive results from 30 seeds. They do not establish statistical superiority.
- This 40-item stress check is separate from the 100-item DQN benchmark in `PHASE_3_EVALUATION.md`.

## Validation boundary

The test suite contains 16 automated tests. The Phase 1 testing plan describes a broader set of 38 cases. The remaining edge cases still require implementation and verification. Do not claim complete formal verification based on this report.
