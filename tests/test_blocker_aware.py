import numpy as np
from baselines.blocker_aware import BlockerAwarePlacement


def test_blocker_aware_avoids_stack_with_earlier_item_below():
    grid=np.zeros((3,3,6),dtype=np.float32)
    grid[0,0]=[1,0.1,0.2,0,0,0]
    summaries=np.asarray([[1/3,2/3,1],[0,1,1],[1/3,2/3,1]],dtype=np.float32)
    obs={'grid':grid,'incoming_item':np.asarray([0.8,0.5,0,1],dtype=np.float32),
         'stack_summaries':summaries,'temp_storage':np.asarray([0,-1,-1],dtype=np.float32)}
    mask=np.asarray([1,1,0],dtype=np.int8)
    assert BlockerAwarePlacement().select_action(obs,mask)==1


def test_blocker_aware_respects_mask_and_unknown_times():
    grid=np.zeros((3,3,6),dtype=np.float32)
    grid[0,0]=[1,-1,0.2,0,0,0]
    obs={'grid':grid,'incoming_item':np.asarray([-1,0.5,0,1],dtype=np.float32),
         'stack_summaries':np.asarray([[1/3,2/3,1],[0,1,1],[0,1,1]],dtype=np.float32),
         'temp_storage':np.asarray([0,-1,-1],dtype=np.float32)}
    mask=np.asarray([0,1,1],dtype=np.int8)
    assert BlockerAwarePlacement().select_action(obs,mask)==1
