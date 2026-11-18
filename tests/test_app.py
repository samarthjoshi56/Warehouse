from app import POLICY_NAMES, run_demo

def test_gradio_demo_runs_baseline_episode():
    state, metrics = run_demo("Improved Greedy", "C", 10, 123)
    assert state.startswith("Warehouse time=")
    assert metrics["retrieved"] == 10
    assert metrics["failed"] == 0
    assert metrics["unplaced"] == 0

def test_dqn_is_listed_only_when_checkpoint_exists():
    from app import CHECKPOINT
    assert ("Trained DQN" in POLICY_NAMES) == CHECKPOINT.exists()
