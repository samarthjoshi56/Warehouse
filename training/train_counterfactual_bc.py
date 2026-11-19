"""Masked counterfactual imitation with episode holdout."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from agents.dqn import QNetwork
from training.episode_split import episode_holdout_mask
def train(data_path,output,epochs=50,seed=0,hidden=128,holdout_seeds=2):
 d=np.load(data_path);X=d['states'].astype(np.float32);masks=d['action_masks'].astype(bool);y=d['labels'].astype(np.int64);test=episode_holdout_mask(d['scenario_ids'],d['seeds'],holdout_seeds);tr=np.flatnonzero(~test);va=np.flatnonzero(test);rng=np.random.default_rng(seed);torch.manual_seed(seed);model=QNetwork(X.shape[1],masks.shape[1],hidden).cpu();opt=torch.optim.Adam(model.parameters(),lr=1e-3);x=torch.tensor(X);yt=torch.tensor(y);m=torch.tensor(masks);hist=[]
 for ep in range(epochs):
  ix=rng.permutation(tr);model.train();losses=[]
  for st in range(0,len(ix),32):
   b=ix[st:st+32];logits=model(x[b]).masked_fill(~m[b],-1e9);loss=nn.functional.cross_entropy(logits,yt[b]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
  model.eval()
  with torch.no_grad():acc=float((model(x[va]).masked_fill(~m[va],-1e9).argmax(1)==yt[va]).float().mean())
  hist.append({'epoch':ep+1,'loss':float(np.mean(losses)),'validation_accuracy':acc})
 out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);te=len(set(zip(d['scenario_ids'][test].tolist(),d['seeds'][test].tolist())));tre=len(set(zip(d['scenario_ids'][~test].tolist(),d['seeds'][~test].tolist())))
 torch.save({'state_dict':model.state_dict(),'input_dim':X.shape[1],'action_dim':masks.shape[1],'hidden_dim':hidden,'source':str(data_path),'samples':len(y),'history':hist,'train_episodes':tre,'validation_episodes':te},out)
 return {'samples':len(y),'train_samples':len(tr),'validation_samples':len(va),'train_episodes':tre,'validation_episodes':te,'last':hist[-1],'output':str(out)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='artifacts/counterfactual_labels_pilot.npz');p.add_argument('--output',default='artifacts/counterfactual_bc.pt');p.add_argument('--epochs',type=int,default=50);p.add_argument('--seed',type=int,default=0);p.add_argument('--holdout-seeds',type=int,default=2);a=p.parse_args();print(json.dumps(train(a.data,a.output,a.epochs,a.seed,holdout_seeds=a.holdout_seeds),indent=2))
if __name__=='__main__':main()
