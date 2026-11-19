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
