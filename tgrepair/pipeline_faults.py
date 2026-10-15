"""Independent recording-pipeline faults for robustness development.

These declared mechanisms are stress tests, not estimated real incident rates.
They share the canonical export adapter, but not the original per-link injector.
"""
from collections import defaultdict
import math
import random

from .data import Event,Graph,anonymize


def pipeline_faults(graph,seed,target_types,outage_fraction=.1,stale_entity_fraction=.25,namespace_fraction=.25):
    for value in [outage_fraction,stale_entity_fraction,namespace_fraction]:
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Fault fractions must be finite and in [0,1]')
    rng=random.Random(seed+67103);objects=dict(graph.objects);identity={o:o for o in objects}
    histories=defaultdict(list)
    for event in graph.events:
        for obj in event.participants:
            if objects[obj] in target_types:histories[obj].append(event)
    # A fixed route by activity emulates two independently named source systems.
    # Both names may be active over overlapping periods.
    channels={kind:rng.randrange(2) for kind in sorted({e.kind for e in graph.events})}
    aliases={};fragmentations=[]
    for obj,events in sorted(histories.items()):
        if len(events)>=4 and rng.random()<namespace_fraction and {channels[e.kind] for e in events}=={0,1}:
            alias='temporary_pipeline_alias_'+obj
            if alias in objects:raise ValueError('Alias collision')
            objects[alias]=objects[obj];identity[alias]=obj;aliases[obj]=alias;fragmentations.append((obj,alias))
    rows=[]
    for event in graph.events:
        ids=tuple(aliases.get(o,o) if channels[event.kind] else o for o in event.participants)
        rows.append(Event(event.uid,event.time,event.kind,ids,dict(event.attributes)))
    # Outage boundaries are time intervals, so simultaneous records move together.
    outage=None
    if rows and outage_fraction:
        width=max(1,int(len(rows)*outage_fraction));start=rng.randrange(len(rows)-width+1)
        outage=(rows[start].time,rows[start+width-1].time)
    missing=[];survivors=[]
    for event in rows:
        ids=[]
        for obj in event.participants:
            if objects[obj] in target_types and outage is not None and outage[0]<=event.time<=outage[1]:
                missing.append((event.uid,obj))
            else:ids.append(obj)
        survivors.append(Event(event.uid,event.time,event.kind,tuple(ids),dict(event.attributes)))
    # A stale join-table entry persists over a middle block of an object's history.
    # The replacement is held fixed, unlike independent random edge errors.
    observed_histories=defaultdict(list);by_type=defaultdict(list)
    for obj,typ in objects.items():by_type[typ].append(obj)
    for event in survivors:
        for obj in event.participants:
            if objects[obj] in target_types:observed_histories[obj].append(event.uid)
    replacements={}
    for obj,eids in sorted(observed_histories.items()):
        if len(eids)<3 or rng.random()>=stale_entity_fraction:continue
        options=[o for o in by_type[objects[obj]] if identity[o]!=identity[obj]]
        if not options:continue
        replacement=rng.choice(options);width=max(1,len(eids)//2);start=rng.randrange(len(eids)-width+1)
        for eid in eids[start:start+width]:replacements[eid,obj]=replacement
    modified=[];wrong=[]
    for event in survivors:
        ids=list(event.participants)
        for i,obj in enumerate(ids):
            replacement=replacements.get((event.uid,obj))
            if replacement is not None and replacement not in ids:
                ids[i]=replacement;wrong.append((event.uid,obj,replacement))
        modified.append(Event(event.uid,event.time,event.kind,tuple(ids),dict(event.attributes)))
    observed,omap,emap=anonymize(Graph(graph.name+'_pipeline',objects,modified),seed+88493,return_maps=True)
    reference={'event_participation':{emap[e.uid]:[omap[o] for o in e.participants] for e in graph.events},
               'identity':{omap[o]:omap[identity[o]] for o in objects},
               'event_origin':{emap[e.uid]:e.uid for e in graph.events},
               'missing':[(emap[e],omap[o]) for e,o in missing],
               'wrong':[(emap[e],omap[a],omap[b]) for e,a,b in wrong],
               'fragmentations':[(omap[a],omap[b]) for a,b in fragmentations],
               'config':{'seed':seed,'family':'independent_recording_pipeline','outage_fraction':outage_fraction,
                         'stale_entity_fraction':stale_entity_fraction,'namespace_fraction':namespace_fraction,
                         'target_types':sorted(target_types)},
               'outage_interval':outage}
    return observed,reference
