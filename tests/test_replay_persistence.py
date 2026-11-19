import numpy as np
import pytest
from training.replay_buffer import ReplayBuffer

def test_replay_save_load_preserves_data_and_sampling_rng(tmp_path):
    buffer = ReplayBuffer(4, 3, 2, seed=11)
    for i in range(3):
        buffer.add(np.full(3, i, dtype=np.float32), i % 2, float(i),
                   np.full(3, i + 1, dtype=np.float32), i == 2, [True, i % 2 == 0])
    path = tmp_path / "replay.npz"
    buffer.save(path)
    expected = buffer.sample(2)

    restored = ReplayBuffer(4, 3, 2, seed=99)
    restored.load(path)
    actual = restored.sample(2)
    assert len(restored) == 3
    assert restored.position == buffer.position
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])

def test_replay_load_rejects_capacity_mismatch(tmp_path):
    ReplayBuffer(4, 3, 2).save(tmp_path / "replay.npz")
    with pytest.raises(ValueError, match="capacity mismatch"):
        ReplayBuffer(5, 3, 2).load(tmp_path / "replay.npz")

def test_partial_checkpoint_stores_only_occupied_transitions(tmp_path):
    buffer = ReplayBuffer(10, 3, 2, seed=5)
    for i in range(2):
        buffer.add(np.full(3, i, dtype=np.float32), i, float(i),
                   np.full(3, i + 1, dtype=np.float32), False, [True, False])
    path = tmp_path / "partial.npz"
    buffer.save(path)
    with np.load(path, allow_pickle=False) as data:
        assert int(data["stored"]) == 2
        assert data["states"].shape == (2, 3)
        assert data["next_masks"].shape == (2, 2)


def test_wrapped_full_checkpoint_preserves_ring_position_and_data(tmp_path):
    buffer = ReplayBuffer(4, 2, 2, seed=7)
    for i in range(6):
        buffer.add(np.full(2, i, dtype=np.float32), i % 2, float(i),
                   np.full(2, i + 1, dtype=np.float32), False, [True, False])
    path = tmp_path / "wrapped.npz"
    buffer.save(path)
    restored = ReplayBuffer(4, 2, 2, seed=99)
    restored.load(path)
    assert len(restored) == 4
    assert restored.position == 2
    np.testing.assert_array_equal(restored.states, buffer.states)
    np.testing.assert_array_equal(restored.next_states, buffer.next_states)
    np.testing.assert_array_equal(restored.actions, buffer.actions)
    np.testing.assert_array_equal(restored.rewards, buffer.rewards)


def test_loads_legacy_full_capacity_checkpoint(tmp_path):
    buffer = ReplayBuffer(4, 2, 2, seed=4)
    buffer.add(np.array([1, 2], dtype=np.float32), 1, 2.0,
               np.array([3, 4], dtype=np.float32), False, [True, False])
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path, capacity=np.asarray(4), position=np.asarray(1), size=np.asarray(1),
        states=buffer.states, next_states=buffer.next_states, actions=buffer.actions,
        rewards=buffer.rewards, dones=buffer.dones, next_masks=buffer.next_masks,
        rng_state=np.asarray(__import__("json").dumps(buffer.rng.bit_generator.state)))
    restored = ReplayBuffer(4, 2, 2, seed=8)
    restored.load(path)
    assert len(restored) == 1
    np.testing.assert_array_equal(restored.states[0], [1, 2])
    assert not restored.states[1:].any()
