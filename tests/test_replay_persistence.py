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
