"""Evaluate an unchanged feature model on a declared calibration collection."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.action_features import OPERATIONS
from tgrepair.data import sha256
from tgrepair.provenance import snapshot_sources
from train_edit_feature_policy import load


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True);parser.add_argument('--validation',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    destination=ROOT/'reports'/f'{args.output}.json'
    cache=ROOT/'data/edit_learning'/f'{args.output}_validation_scores.npz'
    if destination.exists() or cache.exists():raise FileExistsError(destination)
    model_path=ROOT/'checkpoints'/f'{args.model}.joblib'
    models=joblib.load(model_path);x,y,ops,manifest=load(args.validation)
    scores=np.zeros(len(y));metrics={}
    with threadpool_limits(limits=4):
        for op,name in enumerate(OPERATIONS):
            mask=ops==op;scores[mask]=models[op].predict_proba(x[mask])[:,1]
            metrics[name]={'average_precision':float(average_precision_score(y[mask],scores[mask])),
                           'actions':int(mask.sum()),'positives':int(y[mask].sum())}
    np.savez_compressed(cache,probability=scores,label=y,operation=ops)
    report={'status':'validation_score_cache_only_no_fit','config':vars(args),'source_snapshot':snapshot_sources(ROOT),
            'model_hash':sha256(model_path),'validation_manifest_hash':sha256(ROOT/'data/edit_learning'/args.validation/'manifest.json'),
            'score_cache_hash':sha256(cache),'metrics':metrics,
            'contract':'Model parameters are unchanged. This cache permits an explicit matched-calibration control.'}
    destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
