# Phase 2: Training Infrastructure

This package contains RL training components.

**Status:** Placeholder — not implemented in Phase 1.

## Planned contents

- `loop.py` — Training loop
- `replay_buffer.py` — Experience replay (DQN)
- `rollout_buffer.py` — On-policy rollout collection (PPO)
- `checkpointer.py` — Model checkpoint management
- `logger.py` — TensorBoard / CSV logging

## Constraints

- Imports from `environment/` and `agents/` only.
- Never imports from `baselines/`, `evaluation/`, or `visualization/`.
- Never modifies `WarehouseEnv` internals.
