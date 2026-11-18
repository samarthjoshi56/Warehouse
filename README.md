# Smart Warehouse Storage Optimisation Agent

A Reinforcement Learning system for minimising reshuffling operations in a stack-based LIFO warehouse.

> **Status:** Phase 1 complete — Problem definition and system design.  
> Phase 2 (simulator implementation) pending approval.

---

## Project Structure

```
smart-warehouse-agent/
├── environment/        # Warehouse simulator (algorithm-agnostic)
├── agents/             # RL agent interfaces
├── baselines/          # Deterministic baseline algorithms
├── training/           # RL training infrastructure (Phase 2+)
├── evaluation/         # Experiment runner and statistical reporter
├── visualization/      # Rendering and charts (Phase 3+)
├── tests/              # Pytest test suite
├── docs/               # Phase 1 documentation
├── app.py              # Hugging Face Spaces entry point (Phase 3+)
└── requirements.txt
```

## Documentation

| Document | Description |
|---|---|
| [PROBLEM_DEFINITION.md](docs/PROBLEM_DEFINITION.md) | Problem statement, scope, glossary |
| [MATHEMATICAL_MODEL.md](docs/MATHEMATICAL_MODEL.md) | MDP formalisation, state/action/reward math |
| [STATE_ACTION_REWARD_DESIGN.md](docs/STATE_ACTION_REWARD_DESIGN.md) | Representation comparison, action masking, reward design |
| [WAREHOUSE_RULES.md](docs/WAREHOUSE_RULES.md) | LIFO rules, reshuffling, temp storage, invariants |
| [BASELINE_ALGORITHMS.md](docs/BASELINE_ALGORITHMS.md) | 5 baseline algorithms with decision logic |
| [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md) | Unit, integration, and edge case test specs |
| [EXPERIMENTAL_DESIGN.md](docs/EXPERIMENTAL_DESIGN.md) | Seeds, splits, metrics, statistical reporting |
| [SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) | Module responsibilities and dependency graph |
| [PHASE_1_REVIEW.md](docs/PHASE_1_REVIEW.md) | Decisions, assumptions, risks, next steps |

## Quick Start (Phase 2+)

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Warehouse Configuration

| Parameter | Default |
|---|---:|
| Stacks | 10 |
| Stack height | 5 |
| Capacity | 50 |
| Items per episode | 100 |
| Temp storage slots | 10 |

## Deployment

Target: [Hugging Face Spaces](https://huggingface.co/spaces) (Phase 3).

---

*Senior RL Engineering project — Phase 1: 2026-11-18*
