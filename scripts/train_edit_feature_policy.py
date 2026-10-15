"""Matched-candidate gradient-boosted edit control, without graph messages."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.action_features import OPERATIONS
from tgrepair.provenance import snapshot_sources
from tgrepair.data import sha256


def load(name):
    folder=ROOT/'data/edit_learning'/name
    manifest=json.loads((folder/'manifest.json').read_text());xs=[];ys=[];ops=[]
    for r in manifest['records']:
        row=torch.load(folder/r['file'],map_location='cpu',weights_only=True)
        xs.append(row['observation']['action_features'].numpy());ys.append(row['labels'].numpy());ops.append(row['observation']['operations'].numpy())
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ops),manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    start=time.monotonic();snapshot=snapshot_sources(ROOT)
    x,y,ops,train=load(args.train);vx,vy,vops,validation=load(args.validation)
    if {r['group'] for r in train['records']} & {r['group'] for r in validation['records']}:raise ValueError('Execution group leakage')
    models=[];metrics={};pred=np.zeros(len(vy))
    with threadpool_limits(limits=8):
        for op,name in enumerate(OPERATIONS):
            mask=ops==op;labels=y[mask]
            model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=0)
            weights=np.where(labels,.5/labels.mean(),.5/(1-labels.mean()))
            model.fit(x[mask],labels,sample_weight=weights);models.append(model)
            vm=vops==op;pred[vm]=model.predict_proba(vx[vm])[:,1]
            metrics[name]={'average_precision':float(average_precision_score(vy[vm],pred[vm])),
                           'roc_auc':float(roc_auc_score(vy[vm],pred[vm])),'positives':int(vy[vm].sum()),'actions':int(vm.sum())}
            print(json.dumps({'operation':name,'validation':metrics[name],'seconds':time.monotonic()-start}),flush=True)
    checkpoint=ROOT/'checkpoints'/f'{args.output}.joblib';joblib.dump(models,checkpoint)
    np.savez_compressed(ROOT/'data/edit_learning'/f'{args.output}_validation_scores.npz',probability=pred,label=vy,operation=vops)
    report={'status':'development_feature_edit_control','config':vars(args),'source_snapshot':snapshot,
            'validation':metrics,'seconds':time.monotonic()-start,'checkpoint_hash':sha256(checkpoint),
            'parameters':models[0].get_params()}
    destination.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
