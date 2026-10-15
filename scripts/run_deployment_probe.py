"""Mixed-event development baseline without an oracle corruption mask.

Bounded single-participant Application role in BPI 2012. Reference identity
equivalences are labels only. This does not itself merge aliases.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph, sha256
from tgrepair.corruption import mixed_faults
from tgrepair.matching import FEATURE_NAMES, ObservedIndex


def prepare(graph, seed, n_queries, budget):
    observed, reference = mixed_faults(graph,seed,{'Application'},missing_rate=.1,wrong_rate=.1,fragment_rate=.25)
    index = ObservedIndex(observed)
    events = list(observed.events)
    random.Random(seed + 8567).shuffle(events)
    events = events[:n_queries]
    arrays, labels, offsets, current, clean, event_kinds = [], [], [0], [], [], []
    for e in events:
        candidates = index.candidates(e.uid,'Application',budget,allow_current=True)
        f = index.features(e.uid,candidates,exclude_query=True)
        presence = np.asarray([o in e.participants for o in candidates],dtype=np.float32)
        arrays.append(np.column_stack([f,presence]))
        truth = {o for o in reference['event_participation'][e.uid] if observed.objects[o]=='Application'}
        assert len(truth)==1
        y = [reference['identity'][o] in truth for o in candidates]
        labels.extend(y)
        offsets.append(offsets[-1]+len(candidates))
        original = [o for o in e.participants if observed.objects[o]=='Application']
        assert len(original)<=1
        curr = candidates.index(original[0]) if original else -1
        current.append(curr)
        clean.append(curr>=0 and y[curr])
        event_kinds.append(e.kind)
    cache = {'x':np.concatenate(arrays),'y':np.asarray(labels,dtype=bool),
             'offsets':np.asarray(offsets,dtype=np.int64),'current':np.asarray(current,dtype=np.int64),
             'clean':np.asarray(clean,dtype=bool),'kinds':np.asarray(event_kinds)}
    info = {'events':len(events),'clean_relations':sum(clean),
            'missing':sum(c<0 for c in current),'wrong':sum(c>=0 and not good for c,good in zip(current,clean)),
            'fault_config':reference['config'],'fragmentations':len(reference['fragmentations']),
            'attribute_fields':sorted({k for e in observed.events for k in e.attributes})}
    return cache, info


def score(cache, model):
    probabilities = model.predict_proba(cache['x'])[:,1]
    comparisons = []
    for threshold in [0.,.5,.8,.95]:
        for change_margin in [0.,.25,.5,.75]:
            counters = Counter()
            for i, current in enumerate(cache['current']):
                start,end = cache['offsets'][i:i+2]
                values = probabilities[start:end]
                labels = cache['y'][start:end]
                counters['candidate_hit'] += bool(labels.any())
                winner = int(values.argmax())
                # Tied maxima retain current if possible; otherwise abstain.
                tied = np.flatnonzero(np.isclose(values,values[winner],rtol=1e-7,atol=1e-8))
                prediction = current
                if len(tied)==1 and values[winner]>=threshold:
                    if current<0 or values[winner]-values[current]>change_margin:
                        prediction = winner
                correct = prediction>=0 and bool(labels[prediction])
                counters['correct'] += correct
                counters['changed'] += prediction!=current
                counters['abstained_missing'] += prediction<0
                if cache['clean'][i]:
                    counters['clean_total'] += 1
                    counters['clean_preserved'] += correct
                else:
                    counters['damaged_total'] += 1
                    counters['damaged_repaired'] += correct
                if current>=0 and not cache['clean'][i]:
                    counters['wrong_total'] += 1
                    counters['wrong_repaired'] += correct
                if current<0:
                    counters['missing_total'] += 1
                    counters['missing_repaired'] += correct
            n=len(cache['current'])
            comparisons.append({'threshold':threshold,'change_margin':change_margin,
                                'accuracy':counters['correct']/n,
                                'candidate_recall':counters['candidate_hit']/n,
                                'clean_preservation':counters['clean_preserved']/max(1,counters['clean_total']),
                                'damaged_recovery':counters['damaged_repaired']/max(1,counters['damaged_total']),
                                'missing_recovery':counters['missing_repaired']/max(1,counters['missing_total']),
                                'wrong_recovery':counters['wrong_repaired']/max(1,counters['wrong_total']),
                                'counts':{k:int(v) for k,v in counters.items()}})
    return comparisons


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--queries',type=int,default=2000)
    p.add_argument('--budget',type=int,default=128)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',default='deployment_matching_v1')
    args=p.parse_args()
    dest=ROOT/'reports'/f'{args.output}.json'
    if dest.exists(): raise FileExistsError(dest)
    cache_dir=ROOT/'data/probe_cache'/args.output
    cache_dir.mkdir(parents=True)
    start=time.monotonic()
    report={'status':'development_mixed_event_baseline','config':vars(args),
            'features':FEATURE_NAMES+['current_assignment'],
            'code_hashes':{str(p.relative_to(ROOT)):sha256(p) for folder in ('tgrepair','scripts') for p in sorted((ROOT/folder).glob('*.py'))},
            'limits':['No oracle corruption location supplied to scorer.',
                      'Only Application participation, whose schema has one entity per event, is scored.',
                      'Alias-equivalent participation is accepted; this baseline does not repair identity clusters.',
                      'Threshold grid is development tuning, not final target evaluation.'], 'data':{}}
    caches={}
    for split in ['fit','tune']:
        path=ROOT/'data/development'/f'bpi2012_{split}_attributes.json.gz'
        caches[split],info=prepare(load_graph(path),args.seed,args.queries,args.budget)
        np.savez_compressed(cache_dir/f'{split}.npz',**caches[split])
        report['data'][split]={**info,'graph_hash':sha256(path)}
        dest.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'prepared':split,**info}),flush=True)
    train=caches['fit']
    y=train['y']
    weights=np.where(y,.5/y.mean(),.5/(1-y.mean()))
    model=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1.,
                                        early_stopping=False,random_state=args.seed)
    with threadpool_limits(limits=8):
        model.fit(train['x'],y,sample_weight=weights)
        report['tune']=score(caches['tune'],model)
    report['no_change_accuracy']=float(caches['tune']['clean'].mean())
    feasible=[r for r in report['tune'] if r['clean_preservation']>=.99]
    report['selected_development_setting']=max(feasible,key=lambda r:r['accuracy']) if feasible else None
    report['seconds']=time.monotonic()-start
    checkpoint=ROOT/'checkpoints'/f'{args.output}.joblib'
    joblib.dump(model,checkpoint)
    report['model_hash']=sha256(checkpoint)
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'no_change_accuracy':report['no_change_accuracy'],
                      'selected_development_setting':report['selected_development_setting'],
                      'seconds':report['seconds']}),flush=True)


if __name__=='__main__': main()
