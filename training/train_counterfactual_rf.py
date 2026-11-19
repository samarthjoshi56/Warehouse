"""RF imitation with scenario-wise episode holdout (no decision leakage)."""
import argparse,json
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import RandomForestClassifier
from training.episode_split import episode_holdout_mask
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='artifacts/counterfactual_labels_10seed.npz');p.add_argument('--output',default='artifacts/counterfactual_rf.joblib');p.add_argument('--trees',type=int,default=300);p.add_argument('--depth',type=int,default=6);p.add_argument('--min-leaf',type=int,default=4);p.add_argument('--holdout-seeds',type=int,default=2);a=p.parse_args()
 d=np.load(a.data);X=d['states'];y=d['labels'];m=d['action_masks'].astype(bool);te=episode_holdout_mask(d['scenario_ids'],d['seeds'],a.holdout_seeds);tr=np.flatnonzero(~te);va=np.flatnonzero(te)
 model=RandomForestClassifier(n_estimators=a.trees,max_depth=a.depth,min_samples_leaf=a.min_leaf,class_weight='balanced',random_state=0,n_jobs=-1);model.fit(X[tr],y[tr]);pred=model.predict(X[va]);probs=model.predict_proba(X[va]);cls=model.classes_.astype(int);masked=[]
 for i,pb in enumerate(probs):
  z=np.full(m.shape[1],-1.);z[cls]=pb;z[~m[va[i]]]=-np.inf;masked.append(z.argmax())
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);joblib.dump({'model':model,'action_dim':m.shape[1],'input_dim':X.shape[1]},out)
 print(json.dumps({'samples':len(y),'train':len(tr),'test':len(va),'episode_train':len(set(zip(d['scenario_ids'][~te].tolist(),d['seeds'][~te].tolist()))),'episode_test':len(set(zip(d['scenario_ids'][te].tolist(),d['seeds'][te].tolist()))),'accuracy':float(np.mean(pred==y[va])),'masked_accuracy':float(np.mean(np.array(masked)==y[va])),'output':str(out)},indent=2))
if __name__=='__main__':main()
