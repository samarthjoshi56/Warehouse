"""Train a random forest classifier from counterfactual action labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',default='artifacts/counterfactual_labels_10seed.npz'); p.add_argument('--output',default='artifacts/counterfactual_rf.joblib'); p.add_argument('--trees',type=int,default=300); p.add_argument('--depth',type=int,default=6); p.add_argument('--min-leaf',type=int,default=4); args=p.parse_args()
 d=np.load(args.data); X=d['states']; y=d['labels']; masks=d['action_masks'].astype(bool)
 tr,va=train_test_split(np.arange(len(y)),test_size=.2,random_state=0,stratify=y)
 model=RandomForestClassifier(n_estimators=args.trees,max_depth=args.depth,min_samples_leaf=args.min_leaf,class_weight='balanced',random_state=0,n_jobs=-1)
 model.fit(X[tr],y[tr]); pred=model.predict(X[va]); acc=float(np.mean(pred==y[va]))
 probs=model.predict_proba(X[va]); cls=model.classes_.astype(int); masked=[]
 for i,pb in enumerate(probs):
  scores=np.full(masks.shape[1],-1.0); scores[cls]=pb; scores[~masks[va[i]]]=-np.inf; masked.append(int(scores.argmax()))
 masked_acc=float(np.mean(np.asarray(masked)==y[va]))
 out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); joblib.dump({'model':model,'action_dim':masks.shape[1],'input_dim':X.shape[1]},out)
 print(json.dumps({'samples':len(y),'train':len(tr),'validation':len(va),'accuracy':acc,'masked_accuracy':masked_acc,'output':str(out)},indent=2))
if __name__=='__main__': main()
