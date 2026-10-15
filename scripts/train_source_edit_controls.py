"""Compare fixed real-history annotation budgets, with and without generated data."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.action_features import OPERATIONS
from tgrepair.provenance import snapshot_sources
from tgrepair.data import sha256
from train_edit_feature_policy import load


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-cache',default='source_edit_controls_v1')
    p.add_argument('--synthetic-cache',default='edit_train_v2');p.add_argument('--output',default='source_edit_learning_v1');args=p.parse_args()
    destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    start=time.monotonic();snapshot=snapshot_sources(ROOT)
    sx,sy,sop,synthetic=load(args.synthetic_cache)
    source=ROOT/'data/edit_learning'/args.source_cache;valid=np.load(source/'validation.npz')
    vx,vy,vop=valid['x'],valid['y'],valid['operation']
    report={'status':'development_annotation_budget_comparison','config':vars(args),'source_snapshot':snapshot,'models':[]}
    with threadpool_limits(limits=8):
        for budget in [16,64,256]:
            cache=np.load(source/f'train_{budget}.npz');rx,ry,rop=cache['x'],cache['y'],cache['operation']
            for regime in ['source_only','mixed']:
                models=[];metrics={};prediction=np.zeros(len(vy))
                for op,name in enumerate(OPERATIONS):
                    rm=rop==op;sm=sop==op
                    x,y=rx[rm],ry[rm]
                    weights=np.full(len(y),1/max(1,len(y)))
                    if regime=='mixed':
                        weights=np.concatenate([weights*.5,np.full(sm.sum(),.5/max(1,sm.sum()))])
                        x=np.concatenate([x,sx[sm]]);y=np.concatenate([y,sy[sm]])
                    positive_mass=weights[y>0].sum();negative_mass=weights[y==0].sum()
                    if not positive_mass or not negative_mass:raise ValueError('Annotation budget has an absent class')
                    weights*=np.where(y>0,.5/positive_mass,.5/negative_mass)*len(y)
                    model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=0)
                    model.fit(x,y,sample_weight=weights);models.append(model)
                    vm=vop==op;prediction[vm]=model.predict_proba(vx[vm])[:,1]
                    metrics[name]={'average_precision':float(average_precision_score(vy[vm],prediction[vm])),'source_examples':int(rm.sum()),'source_positive':int(ry[rm].sum())}
                name=f'{args.output}_{regime}_{budget}'
                model_path=ROOT/'checkpoints'/f'{name}.joblib';joblib.dump(models,model_path)
                np.savez_compressed(ROOT/'data/edit_learning'/f'{name}_validation_scores.npz',probability=prediction,label=vy,operation=vop)
                row={'name':name,'budget':budget,'regime':regime,'validation':metrics,'model_hash':sha256(model_path),'seconds':time.monotonic()-start}
                report['models'].append(row);destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':main()
