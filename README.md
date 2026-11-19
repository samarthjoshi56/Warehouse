---
title: Smart Warehouse Storage Optimisation Agent
sdk: gradio
app_file: app.py
python_version: "3.10"
---

# Smart Warehouse Storage Optimisation Agent

A reinforcement-learning project that learns item-placement policies for a stack-based LIFO warehouse. The operational objective is to reduce reshuffling while completing scheduled retrievals.

## Current status

- **Phase 1:** Core design and audit documents are part of the local project archive. The remaining Phase 1 documents are not yet synchronized to this branch.
- **Phase 2:** Warehouse simulator, event handling, temporary storage, restoration, five heuristic baselines, and automated tests implemented.
- **Phase 3:** Masked Double DQN, replay buffer, training loop, checkpointing, and held-out evaluation implemented. A trained checkpoint must be generated locally; no unvalidated weights are committed.
- **Phase 4:** Interactive Gradio application is provided for baseline and optional trained-DQN demonstrations.

## Project structure

```
environment/     Simulator, events, item/stack models, metrics
agents/          Agent interfaces and masked DQN
baselines/       Random, First Fit, ERF, Min Height, Improved Greedy
training/        Replay buffer, DQN training and evaluation
evaluation/     Episode runner and statistical reporting
visualization/   Warehouse visualisation helpers
app.py           Gradio demo entry point
docs/            Phase 1 specifications and design audits
tests/           Automated tests
```

## Phase documentation

- `docs/PHASE_2_IMPLEMENTATION.md` — simulator implementation and validation status.
- `docs/PHASE_2_STRESS_VALIDATION.md` — 450-episode multi-scenario stress check.
- `docs/PHASE_3_EVALUATION.md` — initial DQN test results and limitations.
- `docs/PHASE_4_DEPLOYMENT.md` — Hugging Face Spaces setup and CI deployment.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

## Train DQN

Training uses seeds 0–49 by default, as specified in `docs/EXPERIMENTAL_DESIGN.md`.

```bash
python -m training.train_dqn --episodes 50 --seed-start 0 --scenario B --output artifacts/dqn_warehouse.pt
```

The command writes a PyTorch checkpoint and per-episode CSV log.

## Evaluate on held-out test seeds

```bash
python -m training.evaluate --checkpoint artifacts/dqn_warehouse.pt --scenario B --output artifacts/test_results.csv
```

The default test set is seeds 200–229. The evaluator compares DQN with all five baselines and writes raw and statistical summary CSV files. The primary ranking metric is total reshuffles; report all metrics and confidence intervals. Do not tune on test seeds.

## Scenarios

- `A`: no expected retrieval-time information (POMDP setting).
- `B`: noisy expected retrieval-time information (POMDP setting).
- `C`: known retrieval-time information with a finite lookahead window. The window is an approximation to the complete scheduler state, not a strict Markov-state guarantee.

## Hugging Face Spaces

Set the Space SDK to **Gradio** and use `app.py` as the entry point. Upload a trained checkpoint to `artifacts/dqn_warehouse.pt` to enable the DQN option. Without a checkpoint, the demo runs the heuristic baselines only.

## Experimental research extensions

Counterfactual rollout experiments and learned-label baselines are documented separately. These are research-only and are not enabled in the demo.

- [Counterfactual rollout pilot](docs/COUNTERFACTUAL_PLACEMENT_EXPERIMENT.md)
- [Rollout horizon comparison](docs/COUNTERFACTUAL_HORIZON_COMPARISON.md)
- [Counterfactual label-learning results](docs/COUNTERFACTUAL_LABEL_LEARNING.md)
- [Counterfactual score regression experiment](docs/COUNTERFACTUAL_SCORE_REGRESSION.md)
- [Counterfactual label collection script](training/collect_counterfactual_labels.py)
- [Counterfactual score regression script](training/train_counterfactual_score_regression.py)
- [Research-only dependencies](requirements-research.txt)

Install research dependencies separately when needed:

```bash
pip install -r requirements-research.txt
```

## Important limitations

- The simulator is a research implementation, not a production warehouse control system.
- Feed-forward DQN is a baseline learner for the partially observable A/B scenarios; recurrent policies may be more suitable.
- Training success does not establish superiority. Use the held-out multi-seed evaluation before making performance claims.
- The finite lookahead buffer in Scenario C does not expose the entire future scheduler queue.


## Blocker-Aware research benchmark

A blocker-aware heuristic is available as `baselines.BlockerAwarePlacement`. It remains research-only. Run paired-seed benchmarks with:

```bash
PYTHONPATH=. python training/evaluate_blocker_aware.py --seeds 30 --items 20
```

Results and interpretation are documented in `docs/BLOCKER_AWARE_EXPERIMENT.md`. This policy is not enabled in the Gradio demo because it has not shown robust gains across scenarios and item sizes.
