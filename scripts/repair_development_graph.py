"""Whole-graph sequential matching baseline on development records only."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import Graph, load_graph, save_graph, sha256
from tgrepair.corruption import mixed_faults
from tgrepair.matching import ObservedIndex
from tgrepair.identity_matching import alias_candidates, match_aliases
from tgrepair.metrics import identity_pair_scores, traceability


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='sequential_development_v1')
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    report_path=ROOT/'reports'/f'{args.output}.json'
    if report_path.exists(): raise FileExistsError(report_path)
    out=ROOT/'data'/args.output
    out.mkdir(parents=True)
    source=ROOT/'data/development/bpi2012_tune_attributes.json.gz'
    model_path=ROOT/'checkpoints/deployment_matching_v1.joblib'
    model=joblib.load(model_path)
    observed,reference=mixed_faults(load_graph(source),args.seed,{'Application'},missing_rate=.1,wrong_rate=.1,fragment_rate=.25)
    save_graph(observed,out/'observed.json.gz')
    (out/'reference.json').write_text(json.dumps(reference,separators=(',',':')))
    report={'status':'running_development_sequential_baseline','seed':args.seed,
            'source_hash':sha256(source),'model_hash':sha256(model_path),
            'code_hashes':{str(p.relative_to(ROOT)):sha256(p) for folder in ('tgrepair','scripts') for p in sorted((ROOT/folder).glob('*.py'))},
            'before':traceability(observed,reference,{'Application'}),
            'limits':['BPI 2012 tune is development data.','Participation scorer is source-trained; alias scoring is a hand-designed baseline.',
                      'Inference uses a fixed corrupted graph; identity matching is a second stage.']}
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    index=ObservedIndex(observed)
    result=[]
    counters=Counter()
    start=time.monotonic()
    with threadpool_limits(limits=8):
        for batch_start in range(0,len(observed.events),256):
            events=observed.events[batch_start:batch_start+256]
            arrays,candidate_lists,offsets=[],[],[0]
            for e in events:
                candidates=index.candidates(e.uid,'Application',128,allow_current=True)
                features=index.features(e.uid,candidates,exclude_query=True)
                arrays.append(np.column_stack([features,[o in e.participants for o in candidates]]))
                candidate_lists.append(candidates);offsets.append(offsets[-1]+len(candidates))
            values=model.predict_proba(np.concatenate(arrays))[:,1]
            for i,e in enumerate(events):
                candidates=candidate_lists[i];scores=values[offsets[i]:offsets[i+1]]
                current=[o for o in e.participants if observed.objects[o]=='Application']
                winner=int(scores.argmax())
                tied=np.flatnonzero(np.isclose(scores,scores[winner],rtol=1e-7,atol=1e-8))
                prediction=current[0] if current else None
                if len(tied)==1: prediction=candidates[winner]
                participants=tuple([o for o in e.participants if observed.objects[o]!='Application']+([prediction] if prediction else []))
                result.append(replace(e,participants=participants))
                truth={o for o in reference['event_participation'][e.uid] if observed.objects[o]=='Application'}
                was_correct=bool(current) and reference['identity'][current[0]] in truth
                now_correct=prediction is not None and reference['identity'][prediction] in truth
                counters['originally_correct']+=int(was_correct)
                counters['correct_preserved']+=int(was_correct and now_correct)
                counters['incorrect_recovered']+=int(not was_correct and now_correct)
                counters['correct_after']+=int(now_correct)
                counters['changed']+=int(set(current)!=({prediction} if prediction else set()))
            if batch_start % 4096==0:
                print(json.dumps({'processed':len(result),'total':len(observed.events),'seconds':time.monotonic()-start}),flush=True)
    repaired=Graph(observed.name+'_participation_repaired',dict(observed.objects),result)
    repaired.validate();save_graph(repaired,out/'repaired.json.gz')
    report['after_participation']=traceability(repaired,reference,{'Application'})
    report['counts']=dict(counters)
    report['alias_comparisons']=[]
    edges=alias_candidates(repaired,{'Application'},use_attributes=True)
    for threshold in [.25,.5,.75,.9]:
        for assignment in [False,True]:
            clusters=match_aliases(repaired,edges,{'Application'},threshold,assignment)
            report['alias_comparisons'].append({'threshold':threshold,'global_assignment':assignment,
                                               'traceability':traceability(repaired,reference,{'Application'},clusters),
                                               'identity':identity_pair_scores(repaired,reference,{'Application'},clusters)})
    report['seconds']=time.monotonic()-start
    report['status']='completed_development_sequential_baseline'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'before_f1':report['before']['f1'],'after_participation_f1':report['after_participation']['f1'],
                      'best_after_alias_f1':max(r['traceability']['f1'] for r in report['alias_comparisons']),
                      'counts':dict(counters),'seconds':report['seconds']}),flush=True)


if __name__=='__main__': main()
