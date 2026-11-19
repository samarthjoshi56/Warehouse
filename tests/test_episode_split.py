import numpy as np
import pytest
from training.episode_split import episode_holdout_mask
def test_scenario_episode_split():
 s=np.repeat([0,1,2],3);se=np.tile([1,2,3],3);m=episode_holdout_mask(s,se,1)
 assert m.tolist()==[False,False,True]*3
 assert set(zip(s[~m],se[~m])).isdisjoint(set(zip(s[m],se[m])))
def test_small_group_error():
 with pytest.raises(ValueError):episode_holdout_mask(np.array([0,0]),np.array([1,2]),2)
