"""Nested source-development annotation budgets for matched edit controls."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.action_features import ACTION_DIM
from tgrepair.corruption import mixed_faults
from tgrepair.data import load_graph
from tgrepair.entity_features import EntityIndex
from tgrepair.matching import ObservedIndex
from tgrepair.provenance import snapshot_sources


def build(graph,reference,selected,event_limit,target_types=None):
    target_types=set(target_types or {'Application'})
    index=ObservedIndex(graph);identity=reference['identity'];events=[]
    for event in graph.events:
        truth={o for o in reference['event_participation'][event.uid] if graph.objects[o] in target_types}
        if selected is None or truth&selected:events.append(event)
    annotated_event_count=len(events)
    if len(events)>event_limit:events=random.Random(177).sample(events,event_limit)
    arrays=[];labels=[];ops=[]
    for event in events:
        truth=set(reference['event_participation'][event.uid])
        existing={identity[o] for o in event.participants}
        for typ in sorted(target_types):
            candidates=index.candidates(event.uid,typ,128,allow_current=True)
            rows=index.features(event.uid,candidates,exclude_query=True)
            for obj,row in zip(candidates,rows):
                op=1 if obj in event.participants else 0
                x=np.zeros(ACTION_DIM,dtype=np.float32);x[:len(row)]=row
                x[25]=index.type_kind_count[typ][event.kind]/max(1,index.kind_count[event.kind])
                x[26]=sum(graph.objects[o]==typ for o in event.participants)
                label=(identity[obj] not in truth) if op==1 else (identity[obj] in truth-existing)
                arrays.append(x);labels.append(label);ops.append(op)
    entities=EntityIndex(graph,target_types)
    anchors=([o for o in entities.profile if identity[o] in selected] if selected is not None else None)
    pairs=entities.pairs(64,anchors);retrieved=set(pairs)
    # Complete verified histories include their aliases. Use all available
    # positive training pairs; do not weaken low-budget controls by discarding
    # annotations merely because inference retrieval missed them.
    if selected is not None:
        verified={true:[o for o in anchors if identity[o]==true] for true in selected}
        pairs=sorted(retrieved|{tuple(sorted(pair)) for ids in verified.values() for pair in combinations(ids,2)})
    for a,b in pairs:
        x=np.zeros(ACTION_DIM,dtype=np.float32);x[:25]=entities.features(a,b)
        arrays.append(x);labels.append(identity[a]==identity[b]);ops.append(2)
    return np.clip(np.asarray(arrays),-10,10),np.asarray(labels,dtype=np.float32),np.asarray(ops,dtype=np.int64),{
        'verified_entities':len(selected) if selected is not None else None,'verified_history_events':annotated_event_count,
        'sampled_training_events':len(events),'candidate_pairs':len(pairs),'known_handles':len(anchors) if anchors is not None else None,
        'annotated_positive_pairs_added_for_training':len(pairs)-len(retrieved)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='source_edit_controls_v1');args=p.parse_args()
    out=ROOT/'data/edit_learning'/args.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.monotonic();snapshot=snapshot_sources(ROOT)
    graph,reference=mixed_faults(load_graph(ROOT/'data/development/bpi2012_fit_attributes.json.gz'),0,{'Application'})
    identities=sorted({identity for obj,identity in reference['identity'].items() if graph.objects[obj]=='Application'})
    random.Random(422).shuffle(identities)
    report={'status':'development_source_annotation_controls','source_snapshot':snapshot,'budgets':{},
            'annotation_contract':'A verified entity history supplies all its event memberships and alias handles. Candidate negatives against that verified history are labelable; unrelated entity histories remain unlabelled.',
            'limits':['BPI2012 fit context is available without labels. Only sampled verified histories supply training labels.',
                      'All regimes are assessed and tuned on the previously explored BPI2012 tune split; this is not a claim of a total annotation budget including development selection.',
                      'Amount is a denormalized observation condition, not a native per-event BPI2012 attribute.']}
    for budget in [16,64,256]:
        x,y,ops,counts=build(graph,reference,set(identities[:budget]),4096)
        np.savez_compressed(out/f'train_{budget}.npz',x=x,y=y,operation=ops)
        counts['actions']=len(y);counts['positives']=[int(y[ops==op].sum()) for op in range(3)]
        report['budgets'][str(budget)]=counts
        print(json.dumps({'budget':budget,'counts':counts,'seconds':time.monotonic()-start}),flush=True)
    directory=ROOT/'data/sequential_development_v1'
    graph=load_graph(directory/'observed.json.gz');reference=json.loads((directory/'reference.json').read_text())
    x,y,ops,counts=build(graph,reference,None,2048)
    np.savez_compressed(out/'validation.npz',x=x,y=y,operation=ops);report['validation']=counts
    report['seconds']=time.monotonic()-start
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports'/f'{args.output}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
