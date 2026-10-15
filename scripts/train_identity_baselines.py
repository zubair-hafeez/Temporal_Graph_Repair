"""Learn stronger identity baselines from source-real or generated examples."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,sha256
from tgrepair.corruption import mixed_faults
from tgrepair.entity_features import EntityIndex,PAIR_FEATURE_NAMES
from tgrepair.factory_plugin import learner_graph
from tgrepair.identity_matching import match_aliases
from tgrepair.metrics import traceability,identity_pair_scores
from tgrepair.provenance import snapshot_sources


def examples(graph,reference,target_types,budget=64,max_anchors=None):
    index=EntityIndex(graph,target_types)
    anchors=None
    if max_anchors and len(index.profile)>max_anchors:
        anchors=random.Random(913).sample(sorted(index.profile),max_anchors)
    pairs=index.pairs(budget,anchors)
    x=np.asarray([index.features(a,b) for a,b in pairs],dtype=np.float32).reshape(-1,len(PAIR_FEATURE_NAMES))
    y=np.asarray([reference['identity'][a]==reference['identity'][b] for a,b in pairs],dtype=bool)
    return index,pairs,x,y


def predict_clusters(graph,index,pairs,probabilities,target_types,threshold,assignment):
    edges=[]
    for (a,b),score in zip(pairs,probabilities):
        if index.profile[a]['mid']>index.profile[b]['mid']:a,b=b,a
        edges.append((a,b,float(score)))
    return match_aliases(graph,edges,target_types,threshold,assignment)


def evaluate(graph,reference,target_types,model):
    index,pairs,x,y=examples(graph,reference,target_types)
    probabilities=model.predict_proba(x)[:,1]
    true_pairs=sum(1 for a,b in reference['fragmentations'] if graph.objects[a] in target_types)
    comparisons=[]
    for threshold in [.5,.8,.9,.95,.99,.999]:
        for assignment in [False,True]:
            clusters=predict_clusters(graph,index,pairs,probabilities,target_types,threshold,assignment)
            comparisons.append({'threshold':threshold,'global_assignment':assignment,
                                'traceability':traceability(graph,reference,target_types,clusters),
                                'identity':identity_pair_scores(graph,reference,target_types,clusters)})
    return {'candidate_pairs':len(pairs),'true_pairs':true_pairs,'candidate_positive_pairs':int(y.sum()),
            'candidate_recall':float(y.sum()/true_pairs) if true_pairs else None,
            'no_merge':traceability(graph,reference,target_types),'comparisons':comparisons}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--regime',choices=['source','synthetic'],required=True)
    p.add_argument('--generated',default='workflow_train_v3')
    p.add_argument('--episodes',type=int,default=512)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    start=time.monotonic();snapshot=snapshot_sources(ROOT)
    arrays=[];labels=[];counts=Counter()
    if args.regime=='source':
        graph,reference=mixed_faults(load_graph(ROOT/'data/development/bpi2012_fit_attributes.json.gz'),0,{'Application'})
        _,_,x,y=examples(graph,reference,{'Application'},max_anchors=2000)
        arrays.append(x);labels.append(y);counts['graphs']=1
    else:
        path=ROOT/'data/generated'/args.generated
        manifest=json.loads((path/'dataset_manifest.json').read_text())
        for i,episode in enumerate(manifest['episodes'][:args.episodes]):
            variants=['mixed','clean'] if i%4==0 else ['mixed']
            for variant in variants:
                folder=path/episode['id']/variant
                learner=json.loads((folder/'learner.json').read_text());reference=json.loads((folder/'reference.json').read_text())
                graph=learner_graph(learner);target_types=set(learner['schema']['target_types'])
                _,_,x,y=examples(graph,reference,target_types)
                arrays.append(x);labels.append(y);counts['graphs']+=1
            if i%32==0:print(json.dumps({'processed_episodes':i+1,'seconds':time.monotonic()-start}),flush=True)
    x,y=np.concatenate(arrays),np.concatenate(labels)
    counts.update({'pairs':len(y),'positive_pairs':int(y.sum())})
    cache=ROOT/'data/probe_cache'/args.output;cache.mkdir(parents=True)
    np.savez_compressed(cache/'train.npz',x=x,y=y)
    model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,
                                        early_stopping=False,random_state=0)
    weights=np.where(y,.5/y.mean(),.5/(1-y.mean()))
    with threadpool_limits(limits=8):
        model.fit(x,y,sample_weight=weights)
        checkpoint=ROOT/'checkpoints'/f'{args.output}.joblib';joblib.dump(model,checkpoint)
        report={'status':'development_identity_baseline','config':vars(args),'counts':dict(counts),
                'features':PAIR_FEATURE_NAMES,'source_snapshot':snapshot,'model_hash':sha256(checkpoint),
                'parameters':model.get_params(),'evaluations':{}}
        # These are the previously inspected development graphs, not new targets.
        folder=ROOT/'data/sequential_development_v1'
        reference=json.loads((folder/'reference.json').read_text())
        for name,file in [('raw_mixed','observed.json.gz'),('after_participation','repaired.json.gz')]:
            result=evaluate(load_graph(folder/file),reference,{'Application'},model)
            report['evaluations'][name]=result
            report['seconds']=time.monotonic()-start
            destination.write_text(json.dumps(report,indent=2)+'\n')
            best=max(result['comparisons'],key=lambda r:r['traceability']['f1'])
            print(json.dumps({'regime':args.regime,'condition':name,'candidate_recall':result['candidate_recall'],
                              'no_merge_f1':result['no_merge']['f1'],'best_development_f1':best['traceability']['f1'],
                              'threshold':best['threshold'],'global_assignment':best['global_assignment'],
                              'identity':best['identity'],'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
