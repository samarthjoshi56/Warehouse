"""Five baseline placement strategies."""
import numpy as np
from .base_baseline import BaselineAgent
from environment.exceptions import NoValidPlacementError
def _valid(mask):
    valid=np.flatnonzero(np.asarray(mask,dtype=np.int8))
    if not len(valid): raise NoValidPlacementError("No valid stack is available")
    return valid
class RandomPlacement(BaselineAgent):
    def __init__(self,seed=0): self.rng=np.random.default_rng(seed)
    @property
    def name(self): return "Random"
    def select_action(self,observation,action_mask): return int(self.rng.choice(_valid(action_mask)))
class FirstFit(BaselineAgent):
    @property
    def name(self): return "First Fit"
    def select_action(self,observation,action_mask): return int(_valid(action_mask)[0])
class GreedyMinHeight(BaselineAgent):
    @property
    def name(self): return "Greedy (Min Height)"
    def select_action(self,observation,action_mask):
        valid=_valid(action_mask); heights=observation["stack_summaries"][:,0]
        return int(min(valid,key=lambda k:(heights[k],int(k))))
class EarliestRetrievalFirst(BaselineAgent):
    @property
    def name(self): return "Earliest Retrieval First"
    def select_action(self,observation,action_mask):
        valid=_valid(action_mask); tau=float(observation["incoming_item"][0])
        if tau<0: return int(valid[0])
        grid=observation["grid"]; heights=observation["stack_summaries"][:,0]; scores=[]
        for k in valid:
            occupied=grid[k,:,0]>0.5; known=grid[k,occupied,1]; known=known[known>=0]
            score=float("inf") if not occupied.any() else (float(np.min(known))-tau if len(known) else float("-inf"))
            scores.append((score,1.0-heights[k],-int(k),int(k)))
        return max(scores)[-1]
class ImprovedGreedy(BaselineAgent):
    def __init__(self,alpha=0.4,beta=0.6): self.alpha,self.beta=alpha,beta
    @property
    def name(self): return "Improved Greedy"
    def select_action(self,observation,action_mask):
        valid=_valid(action_mask); tau,priority=map(float,observation["incoming_item"][:2])
        grid=observation["grid"]; heights=observation["stack_summaries"][:,0]; scores=[]
        for k in valid:
            occupied=np.flatnonzero(grid[k,:,0]>0.5); compatibility=1.0
            if len(occupied):
                top=grid[k,occupied[-1]]; tau_top,priority_top=float(top[1]),float(top[2])
                if tau<0 or tau_top<0: compatibility=0.0
                elif tau<tau_top or (tau==tau_top and priority<=priority_top): compatibility=1.0
                else: compatibility=-1.0
            score=-self.alpha*float(heights[k])+self.beta*compatibility
            scores.append((score,-float(heights[k]),-int(k),int(k)))
        return max(scores)[-1]
