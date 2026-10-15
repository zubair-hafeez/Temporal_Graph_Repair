"""Candidate graph edits and graph tensors derived solely from observations."""
from collections import Counter,defaultdict
from dataclasses import dataclass
import math
import random

import numpy as np

from .edits import Edit
from .entity_features import EntityIndex
from .matching import ObservedIndex


OPERATIONS=['ADD_PARTICIPANT','REMOVE_PARTICIPANT','MERGE_ALIASES']
ACTION_DIM=28
NODE_DIM=16


@dataclass
class ActionObservation:
    graph: object
    edits: list
    action_features: np.ndarray
    node_features: np.ndarray
    node_types: np.ndarray
    activity_types: np.ndarray
    edges: np.ndarray
    edge_features: np.ndarray
    action_nodes: np.ndarray
    operations: np.ndarray


def encode_graph(graph,target_types,seed=0):
    rng=random.Random(seed)
    types=sorted(set(graph.objects.values()));type_slots=list(range(1,17));rng.shuffle(type_slots)
    type_map={t:type_slots[i%16] for i,t in enumerate(types)}
    activities=sorted({e.kind for e in graph.events});activity_slots=list(range(1,129));rng.shuffle(activity_slots)
    activity_map={t:activity_slots[i%128] for i,t in enumerate(activities)}
    event_nodes={e.uid:i for i,e in enumerate(graph.events)}
    object_nodes={o:len(event_nodes)+i for i,o in enumerate(sorted(graph.objects))}
    n=len(event_nodes)+len(object_nodes)
    features=np.zeros((n,NODE_DIM),dtype=np.float32)
    type_ids=np.zeros(n,dtype=np.int64);activity_ids=np.zeros(n,dtype=np.int64)
    histories=defaultdict(list);kind_count=Counter(e.kind for e in graph.events)
    for e in graph.events:
        for o in e.participants:histories[o].append(e)
    t0=graph.events[0].time if graph.events else 0.
    span=max(1.,graph.events[-1].time-t0) if graph.events else 1.
    edges=[];edge_features=[]
    for e in graph.events:
        i=event_nodes[e.uid]
        features[i,:8]=[1,0,math.log1p(len(e.participants)),(e.time-t0)/span,
                        kind_count[e.kind]/max(1,len(graph.events)),len(e.attributes),
                        sum(graph.objects[o] in target_types for o in e.participants),0]
        activity_ids[i]=activity_map[e.kind]
        for o in e.participants:
            j=object_nodes[o]
            edges.extend([(i,j),(j,i)])
            edge_features.extend([[0,0,0,float(graph.objects[o] in target_types)],
                                  [1,0,0,float(graph.objects[o] in target_types)]])
    for o,j in object_nodes.items():
        hist=histories[o];typ=graph.objects[o];type_ids[j]=type_map[typ]
        if hist:
            times=np.asarray([e.time for e in hist]);unique_kinds={e.kind for e in hist}
            features[j,:10]=[0,1,math.log1p(len(hist)),(times.mean()-t0)/span,
                             len(unique_kinds)/max(1,len(activities)),0,float(typ in target_types),
                             (times[0]-t0)/span,(times[-1]-t0)/span,float(times.std()/span)]
            groups=[]
            for e in hist:
                if groups and groups[-1][0]==e.time:groups[-1][1].append(e)
                else:groups.append((e.time,[e]))
            for (ta,previous),(tb,following) in zip(groups,groups[1:]):
                # Bounded ties preserve simultaneity; no source row order enters.
                for a in sorted(previous,key=lambda e:e.uid)[:8]:
                    for b in sorted(following,key=lambda e:e.uid)[:8]:
                        gap=math.log1p((tb-ta)/span)
                        same=float(a.kind==b.kind)
                        edges.extend([(event_nodes[a.uid],event_nodes[b.uid]),(event_nodes[b.uid],event_nodes[a.uid])])
                        edge_features.extend([[2,gap,same,float(typ in target_types)],
                                              [3,-gap,same,float(typ in target_types)]])
        else:features[j,:3]=[0,1,0];features[j,6]=float(typ in target_types)
    return features,type_ids,activity_ids,np.asarray(edges,dtype=np.int64).reshape(-1,2).T,np.asarray(edge_features,dtype=np.float32).reshape(-1,4),event_nodes,object_nodes


def propose_actions(state,target_types,seed=0,event_budget=48,participant_budget=16,pair_budget=32):
    graph=state.active_graph();index=ObservedIndex(graph)
    selected=list(graph.events)
    if event_budget is not None and len(selected)>event_budget:
        random.Random(seed+7919).shuffle(selected);selected=selected[:event_budget]
    edits=[];features=[]
    for e in selected:
        for typ in sorted(target_types):
            candidates=index.candidates(e.uid,typ,participant_budget,allow_current=True)
            candidate_features=index.features(e.uid,candidates,exclude_query=True)
            for o,row in zip(candidates,candidate_features):
                operation='REMOVE_PARTICIPANT' if o in e.participants else 'ADD_PARTICIPANT'
                values=np.zeros(ACTION_DIM,dtype=np.float32);values[:len(row)]=row
                values[25]=index.type_kind_count[typ][e.kind]/max(1,index.kind_count[e.kind])
                values[26]=sum(graph.objects[p]==typ for p in e.participants)
                edits.append(Edit(operation,e.uid,o));features.append(values)
    identities=EntityIndex(graph,target_types)
    for a,b in identities.pairs(pair_budget):
        values=np.zeros(ACTION_DIM,dtype=np.float32);values[:25]=identities.features(a,b)
        edits.append(Edit('MERGE_ALIASES',a,b));features.append(values)
    nf,nt,at,edges,ef,event_nodes,object_nodes=encode_graph(graph,target_types,seed)
    action_nodes=[];operations=[]
    for edit in edits:
        source=object_nodes[edit.source] if edit.operation=='MERGE_ALIASES' else event_nodes[edit.source]
        action_nodes.append((source,object_nodes[edit.target]));operations.append(OPERATIONS.index(edit.operation))
    return ActionObservation(graph,edits,np.clip(np.asarray(features,dtype=np.float32).reshape(-1,ACTION_DIM),-10,10),
                             nf,nt,at,edges,ef,np.asarray(action_nodes,dtype=np.int64).reshape(-1,2),np.asarray(operations,dtype=np.int64))
