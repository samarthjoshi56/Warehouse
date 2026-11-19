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
