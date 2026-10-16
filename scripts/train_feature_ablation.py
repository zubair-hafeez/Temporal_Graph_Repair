"""Refit matched scoring controls; retrieval is deliberately held unchanged."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.action_features import ACTION_DIM,OPERATIONS
from tgrepair.data import sha256
from tgrepair.provenance import snapshot_sources
from train_edit_feature_policy import load

MASKS={
    'temporal_scoring':{0:[5,6,7,8,10,11],1:[5,6,7,8,10,11],2:[7,8,9,10,11,21,22,23]},
    'context_scoring':{0:[1,2,3,4,6,12],1:[1,2,3,4,6,12],2:[13,14,15,16]},
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',default='edit_train_attributes_v3')
    parser.add_argument('--validation',default='edit_validation_attributes_v3')
    parser.add_argument('--remove',choices=MASKS,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    start=time.monotonic();snapshot=snapshot_sources(ROOT)
    x,y,ops,train=load(args.train);vx,vy,vops,valid=load(args.validation)
    if {r['group'] for r in train['records']}&{r['group'] for r in valid['records']}:
        raise ValueError('Execution-group leakage')
    models=[];metrics={};prediction=np.zeros(len(vy))
    parameters=dict(max_iter=200,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=0)
    with threadpool_limits(limits=4):
        for op,name in enumerate(OPERATIONS):
            keep=[j for j in range(ACTION_DIM) if j not in MASKS[args.remove][op]]
            estimator=Pipeline([('fields',ColumnTransformer([('retained','passthrough',keep)],remainder='drop')),
                                ('classifier',HistGradientBoostingClassifier(**parameters))])
            mask=ops==op;labels=y[mask]
            weights=np.where(labels,.5/labels.mean(),.5/(1-labels.mean()))
            estimator.fit(x[mask],labels,classifier__sample_weight=weights)
            models.append(estimator)
            vm=vops==op;prediction[vm]=estimator.predict_proba(vx[vm])[:,1]
            metrics[name]={'average_precision':float(average_precision_score(vy[vm],prediction[vm])),
                           'removed_columns':MASKS[args.remove][op],'actions':int(vm.sum()),'positives':int(vy[vm].sum())}
            print(json.dumps({'operation':name,'validation':metrics[name]}),flush=True)
    model_path=ROOT/'checkpoints'/f'{args.output}.joblib'
    joblib.dump(models,model_path)
    np.savez_compressed(ROOT/'data/edit_learning'/f'{args.output}_validation_scores.npz',
                        probability=prediction,label=vy,operation=vops)
    report={'status':'development_refitted_scoring_ablation','config':vars(args),
            'source_snapshot':snapshot,'parameters':parameters,'validation':metrics,
            'checkpoint_hash':sha256(model_path),'seconds':time.monotonic()-start,
            'contract':'The candidate pools and training labels are held fixed. Only classifier inputs are removed and models refitted.',
            'limits':['Temporal and contextual retrieval remains in both controls. These are scoring-feature ablations, not graph-free or time-free systems.']}
    destination.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
