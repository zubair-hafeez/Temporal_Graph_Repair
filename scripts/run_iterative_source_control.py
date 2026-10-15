"""Test whether refreshing strong source-trained matchers explains repair gains."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,save_graph
from tgrepair.edits import RepairState,Edit
from tgrepair.entity_features import EntityIndex
from tgrepair.identity_matching import match_aliases
from tgrepair.matching import ObservedIndex
from tgrepair.metrics import traceability,identity_pair_scores
from tgrepair.provenance import snapshot_sources


def participant_pass(state,model,budget=128):
    graph=state.active_graph();index=ObservedIndex(graph);result=[];changed=0;start=time.monotonic()
    for batch_start in range(0,len(graph.events),512):
        events=graph.events[batch_start:batch_start+512];arrays=[];lists=[];offsets=[0]
        for event in events:
            candidates=index.candidates(event.uid,'Application',budget,allow_current=True)
            features=index.features(event.uid,candidates,exclude_query=True)
            arrays.append(np.column_stack([features,[o in event.participants for o in candidates]]))
            lists.append(candidates);offsets.append(offsets[-1]+len(candidates))
        values=model.predict_proba(np.concatenate(arrays))[:,1]
        for i,event in enumerate(events):
            candidates=lists[i];scores=values[offsets[i]:offsets[i+1]]
            current=[o for o in event.participants if graph.objects[o]=='Application']
            winner=int(scores.argmax());tied=np.flatnonzero(np.isclose(scores,scores[winner],rtol=1e-7,atol=1e-8))
            prediction=candidates[winner] if len(tied)==1 else (current[0] if current else None)
            participants=tuple(sorted([o for o in event.participants if graph.objects[o]!='Application']+([prediction] if prediction else [])))
            changed+=set(participants)!=set(event.participants);result.append(replace(event,participants=participants))
        if batch_start%8192==0:print(json.dumps({'participation_processed':len(result),'seconds':time.monotonic()-start}),flush=True)
    state.graph.events=result
    return int(changed)


def identity_pass(state,model,threshold):
    graph=state.active_graph();index=EntityIndex(graph,{'Application'});pairs=index.pairs(64)
    features=np.asarray([index.features(a,b) for a,b in pairs],dtype=np.float32)
    scores=model.predict_proba(features)[:,1];edges=[]
    for (a,b),score in zip(pairs,scores):
        if index.profile[a]['mid']>index.profile[b]['mid']:a,b=b,a
        edges.append((a,b,float(score)))
    mapping=match_aliases(graph,edges,{'Application'},threshold,False);groups=defaultdict(list)
    for o,c in mapping.items():groups[c].append(o)
    changed=0
    for ids in groups.values():
        ids=sorted(ids)
        for other in ids[1:]:state.apply(Edit('MERGE_ALIASES',ids[0],other));changed+=1
    return changed


def main():
    p=argparse.ArgumentParser();p.add_argument('--identity-model',default='identity_learned_source_v1')
    p.add_argument('--threshold',type=float,default=.9);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--output',required=True);args=p.parse_args()
    destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    snapshot=snapshot_sources(ROOT);start=time.monotonic()
    directory=ROOT/'data/sequential_development_v1'
    state=RepairState(load_graph(directory/'repaired.json.gz'));reference=json.loads((directory/'reference.json').read_text())
    participant_model=joblib.load(ROOT/'checkpoints/deployment_matching_v1.joblib')
    identity_model=joblib.load(ROOT/'checkpoints'/f'{args.identity_model}.joblib')
    report={'status':'development_iterative_source_control','config':vars(args),'source_snapshot':snapshot,
            'initial_after_participation':traceability(state.graph,reference,{'Application'}),'iterations':[],
            'schema_assumption':'Exactly one Application per event, shared with the prior source-trained sequential control.'}
    out=ROOT/'data'/args.output;out.mkdir(parents=True)
    with threadpool_limits(limits=8):
        for iteration in range(args.rounds):
            if iteration:
                changed=participant_pass(state,participant_model)
                report['iterations'].append({'round':iteration+1,'stage':'participation','changes':changed,
                                             'history':traceability(state.graph,reference,{'Application'},state.clusters),
                                             'identity':identity_pair_scores(state.graph,reference,{'Application'},state.clusters)})
            changed=identity_pass(state,identity_model,args.threshold)
            row={'round':iteration+1,'stage':'identity','changes':changed,
                 'history':traceability(state.graph,reference,{'Application'},state.clusters),
                 'identity':identity_pair_scores(state.graph,reference,{'Application'},state.clusters)}
            report['iterations'].append(row);report['seconds']=time.monotonic()-start
            save_graph(state.graph,out/f'round{iteration+1}.json.gz')
            (out/f'round{iteration+1}_clusters.json').write_text(json.dumps(state.clusters)+'\n')
            destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
    print(json.dumps({'seconds':report['seconds'],'final_f1':report['iterations'][-1]['history']['f1']}),flush=True)


if __name__=='__main__':main()
