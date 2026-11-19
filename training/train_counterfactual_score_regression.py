"""Counterfactual score regression with scenario-stratified episode holdout."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from training.episode_split import episode_holdout_mask

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='artifacts/counterfactual_labels_10seed.npz');p.add_argument('--output',default='artifacts/counterfactual_score_model.npz');p.add_argument('--trees',type=int,default=300);p.add_argument('--depth',type=int,default=8);p.add_argument('--min-leaf',type=int,default=3);p.add_argument('--holdout-seeds',type=int,default=2);p.add_argument('--model',choices=['rf','extra','histgb'],default='extra');a=p.parse_args()
 d=np.load(a.data);X=d['states'].astype(np.float32);m=d['action_masks'].astype(bool);s=d['candidate_scores'][:,:,0].astype(np.float32);sc=d['scenario_ids'];seed=d['seeds'];test=episode_holdout_mask(sc,seed,a.holdout_seeds);train=~test
 models={};pred=np.full_like(s,np.nan)
 for act in range(m.shape[1]):
  elig=train&m[:,act]&np.isfinite(s[:,act])
  if elig.sum()<10:continue
  if a.model=='rf':model=RandomForestRegressor(n_estimators=a.trees,max_depth=a.depth,min_samples_leaf=a.min_leaf,random_state=act,n_jobs=-1)
  elif a.model=='extra':model=ExtraTreesRegressor(n_estimators=a.trees,max_depth=a.depth,min_samples_leaf=a.min_leaf,random_state=act,n_jobs=-1)
  else:model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=20,min_samples_leaf=5,random_state=act)
  model.fit(X[elig],s[elig,act]);models[act]=model;pred[test,act]=model.predict(X[test])
 valid=test&m.any(axis=1);vp=np.where(m[valid],pred[valid],np.inf)
 if np.any(~np.isfinite(vp).any(axis=1)):raise RuntimeError('No action prediction available for a held-out decision')
 chosen=vp.argmin(axis=1);oracle=np.where(m[valid],s[valid],np.inf).argmin(axis=1);cs=s[valid,chosen];os=s[valid,oracle];scv=sc[valid]
 metrics={'model':a.model,'samples':len(X),'train_decisions':int(train.sum()),'test_decisions':int(test.sum()),'train_episodes':len(set(zip(sc[train].tolist(),seed[train].tolist()))),'test_episodes':len(set(zip(sc[test].tolist(),seed[test].tolist()))),'test_episode_keys':sorted(set((int(x),int(y)) for x,y in zip(sc[test],seed[test]))),'action_agreement':float(np.mean(chosen==oracle)),'mean_regret':float(np.mean(cs-os)),'mean_selected_score':float(np.mean(cs)),'mean_oracle_score':float(np.mean(os)),'per_scenario':{str(int(x)):{'agreement':float(np.mean(chosen[scv==x]==oracle[scv==x])),'regret':float(np.mean(cs[scv==x]-os[scv==x]))} for x in np.unique(scv)}}
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,models=np.array([models],dtype=object),action_dim=m.shape[1],input_dim=X.shape[1],metrics=np.array([metrics],dtype=object));print(json.dumps(metrics,indent=2));print('saved:',out)
if __name__=='__main__':main()
