"""Inference-only policy trained from counterfactual rollout labels."""
import numpy as np
import torch
from agents.dqn import QNetwork, flatten_observation
class CounterfactualClone:
    def __init__(self,checkpoint,device='cpu'):
        data=torch.load(checkpoint,map_location=device,weights_only=False)
        self.model=QNetwork(data['input_dim'],data['action_dim'],data['hidden_dim']).to(device)
        self.model.load_state_dict(data['state_dict']); self.model.eval(); self.device=device
    def reset(self): pass
    def select_action(self,observation,action_mask=None):
        mask=np.asarray(observation['action_mask'] if action_mask is None else action_mask,dtype=bool)
        if not mask.any(): raise ValueError('No valid action')
        state=torch.as_tensor(flatten_observation(observation),dtype=torch.float32,device=self.device).unsqueeze(0)
        with torch.no_grad(): logits=self.model(state).squeeze(0).cpu().numpy()
        logits[~mask]=-np.inf
        return int(np.argmax(logits))
