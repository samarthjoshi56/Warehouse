import numpy as np
from environment.config import WarehouseConfig
from environment.warehouse import WarehouseEnv
from agents.counterfactual import CounterfactualGreedy

def test_counterfactual_agent_returns_valid_action_and_finishes():
    env=WarehouseEnv(WarehouseConfig(num_items=8,scenario='B'))
    obs,_=env.reset(seed=91)
    agent=CounterfactualGreedy()
    done=False
    while not done:
        mask=obs['action_mask']
        if not mask.any():
            obs,_,term,trunc,_=env.step(0); done=term or trunc; continue
        action=agent.select_action(obs,mask,env=env)
        assert mask[action] == 1
        obs,_,term,trunc,_=env.step(action); done=term or trunc
    metrics=env.get_metrics()
    assert metrics['retrieved'] == 8
    assert metrics['failed'] == 0
    assert metrics['unplaced'] == 0

def test_counterfactual_agent_requires_environment():
    env=WarehouseEnv(WarehouseConfig(num_items=3,scenario='A'))
    obs,_=env.reset(seed=1)
    agent=CounterfactualGreedy()
    try:
        agent.select_action(obs,obs['action_mask'])
    except ValueError as exc:
        assert 'live environment' in str(exc)
    else:
        raise AssertionError('Expected live environment requirement')


def test_counterfactual_collects_action_labels():
    env=WarehouseEnv(WarehouseConfig(num_items=5,scenario='A'))
    obs,_=env.reset(seed=22)
    agent=CounterfactualGreedy(rollout_placements=5,collect_records=True)
    mask=obs['action_mask']
    action=agent.select_action(obs,mask,env=env)
    assert mask[action] == 1
    assert len(agent.records) == 1
    record=agent.records[0]
    assert record['observation'].ndim == 1
    assert record['action_mask'].shape == mask.shape
    assert record['candidate_scores'].shape == (env.action_space.n,3)
    assert np.isfinite(record['candidate_scores'][action]).all()
