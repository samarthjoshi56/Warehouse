import numpy as np
from baselines import RandomPlacement,FirstFit,GreedyMinHeight,EarliestRetrievalFirst,ImprovedGreedy

def observation():
    grid=np.zeros((3,2,6),dtype=np.float32); grid[0,0]=[1,0.5,0.4,0,1,0]
    return {"grid":grid,"incoming_item":np.asarray([0.2,0.2,0,1],dtype=np.float32),
            "stack_summaries":np.asarray([[0.5,0.5,1],[0,1,1],[0.5,0.5,1]],dtype=np.float32)}

def test_baselines_respect_mask():
    obs=observation(); mask=np.asarray([0,1,1],dtype=np.int8)
    agents=[RandomPlacement(4),FirstFit(),GreedyMinHeight(),EarliestRetrievalFirst(),ImprovedGreedy()]
    for agent in agents: assert agent.select_action(obs,mask) in {1,2}

def test_first_fit_is_lowest_valid_index():
    assert FirstFit().select_action(observation(),np.asarray([0,1,1]))==1
