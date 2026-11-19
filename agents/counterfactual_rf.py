"""Inference policy using a random forest trained on counterfactual labels."""
import joblib,numpy as np
from agents.dqn import flatten_observation
class CounterfactualRF:
 def __init__(self,checkpoint):
  data=joblib.load(checkpoint); self.model=data['model']; self.action_dim=data['action_dim']
 def reset(self): pass
 def select_action(self,observation,action_mask=None):
  mask=np.asarray(observation['action_mask'] if action_mask is None else action_mask,dtype=bool)
  if not mask.any(): raise ValueError('No valid action')
  x=flatten_observation(observation).reshape(1,-1); probs=self.model.predict_proba(x)[0]
  scores=np.full(self.action_dim,-np.inf); scores[self.model.classes_.astype(int)]=probs; scores[~mask]=-np.inf
  return int(scores.argmax())
