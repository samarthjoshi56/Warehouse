---
title: Smart Warehouse Storage Optimisation Agent
sdk: gradio
app_file: app.py
python_version: "3.10"
---

# Smart Warehouse Storage Optimisation Agent

A reinforcement-learning project that learns item-placement policies for a stack-based LIFO warehouse. The operational objective is to reduce reshuffling while completing scheduled retrievals.

## Current status

- **Phase 1:** Problem definition and design documents are included in the downloadable project archive.
- **Phase 2:** Warehouse simulator, event handling, temporary storage, restoration, five heuristic baselines, and automated tests implemented.
- **Phase 3:** Masked Double DQN, replay buffer, training loop, checkpointing, and held-out evaluation implemented. Initial DQN results are below the heuristic baselines.
- **Phase 4:** Gradio demo implemented. It runs the five heuristics by default. Add a trained checkpoint at `artifacts/dqn_warehouse.pt` to enable DQN inference.

## Phase documentation

- `docs/PHASE_2_IMPLEMENTATION.md` — simulator implementation and validation status.
- `docs/PHASE_3_EVALUATION.md` — initial DQN test results and limitations.\n- `docs/PHASE_4_DEPLOYMENT.md` — Hugging Face Spaces setup and CI deployment.

## Install

```bash
pip install -r requirements.txt
pytest -q
```

## Train

Training seeds 0–49:

```bash
python -m training.train_dqn --episodes 50 --seed-start 0 --scenario B --output artifacts/dqn_warehouse.pt
```

Resume an interrupted run:

```bash
python -m training.train_dqn --episodes 50 --seed-start 0 --scenario B --output artifacts/dqn_warehouse.pt --resume
```

## Evaluate

Held-out test seeds 200–229:

```bash
python -m training.evaluate --checkpoint artifacts/dqn_warehouse.pt --scenario B --output artifacts/test_results.csv
```

The evaluator writes raw per-seed results and a summary with mean, median, standard deviation, and 95% confidence intervals. See `docs/PHASE_3_EVALUATION.md` for the initial result.

## Important limitations

- This is a research simulator, not a production warehouse controller.
- The initial DQN did not outperform the heuristic baselines.
- Feed-forward DQN is a baseline learner for partially observable Scenarios A/B; recurrent policies may be more suitable.
- The finite lookahead in Scenario C is not a strict Markov-state guarantee.
- Do not make performance claims without multi-seed held-out evaluation.
