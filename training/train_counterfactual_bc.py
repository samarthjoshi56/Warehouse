"""Train a masked classifier from counterfactual rollout labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from agents.dqn import QNetwork

def train(data_path, output, epochs=50, seed=0, hidden=128):
    data=np.load(data_path)
    states=data['states'].astype(np.float32); masks=data['action_masks'].astype(bool); labels=data['labels'].astype(np.int64)
    rng=np.random.default_rng(seed); order=rng.permutation(len(labels)); cut=max(1,int(len(order)*0.8)); train_ix,valid_ix=order[:cut],order[cut:]
    torch.manual_seed(seed); model=QNetwork(states.shape[1],masks.shape[1],hidden).cpu()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); x=torch.tensor(states); y=torch.tensor(labels); m=torch.tensor(masks); history=[]
    for ep in range(epochs):
        shuffled=rng.permutation(train_ix); model.train(); losses=[]
        for start in range(0,len(shuffled),32):
            ix=shuffled[start:start+32]; logits=model(x[ix]).masked_fill(~m[ix],-1e9); loss=nn.functional.cross_entropy(logits,y[ix]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_ix=valid_ix if len(valid_ix) else train_ix
            logits=model(x[val_ix]).masked_fill(~m[val_ix],-1e9)
            accuracy=float((logits.argmax(1)==y[val_ix]).float().mean())
        history.append({'epoch':ep+1,'loss':float(np.mean(losses)),'validation_accuracy':accuracy})
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'input_dim':states.shape[1],'action_dim':masks.shape[1],'hidden_dim':hidden,'source':str(data_path),'samples':len(labels),'history':history},output)
    return {'samples':len(labels),'train_samples':len(train_ix),'validation_samples':len(valid_ix),'last':history[-1],'output':str(output)}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='artifacts/counterfactual_labels_pilot.npz'); p.add_argument('--output',default='artifacts/counterfactual_bc.pt'); p.add_argument('--epochs',type=int,default=50); p.add_argument('--seed',type=int,default=0); args=p.parse_args(); print(json.dumps(train(args.data,args.output,args.epochs,args.seed),indent=2))
if __name__=='__main__': main()
