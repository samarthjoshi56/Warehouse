# RL Training

Train the masked Double DQN with the 50 training seeds (0–49):

```bash
python -m training.train_dqn --episodes 50 --seed-start 0 --scenario B --output artifacts/dqn_warehouse.pt
```

Resume an interrupted run with `--resume`. Evaluate on held-out seeds 200–229:

```bash
python -m training.evaluate --checkpoint artifacts/dqn_warehouse.pt --scenario B
```

The initial DQN did not beat the heuristic baselines. See `docs/PHASE_3_EVALUATION.md`.
