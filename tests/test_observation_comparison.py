from training.compare_observation_versions import environment_seed_start_for_run

def test_environment_seed_blocks_are_disjoint():
    episodes = 100
    starts = [environment_seed_start_for_run(seed, episodes) for seed in range(3)]
    assert starts == [0, 100, 200]
    blocks = [set(range(start, start + episodes)) for start in starts]
    assert blocks[0].isdisjoint(blocks[1])
    assert blocks[1].isdisjoint(blocks[2])
    assert blocks[0].isdisjoint(blocks[2])

def test_seed_block_start_handles_custom_episode_count():
    assert environment_seed_start_for_run(4, 25) == 100


def test_paired_bootstrap_ci_is_deterministic_and_contains_constant_effect():
    from training.compare_observation_versions import paired_bootstrap_ci
    values = [2.0] * 20
    first = paired_bootstrap_ci(values, seed=123)
    second = paired_bootstrap_ci(values, seed=123)
    assert first == second
    assert first == [2.0, 2.0]

def test_paired_bootstrap_ci_rejects_empty_data():
    import pytest
    from training.compare_observation_versions import paired_bootstrap_ci
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], seed=1)
