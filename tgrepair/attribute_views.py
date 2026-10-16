"""Observation-only selection of stable, discriminative entity evidence.

This is an optional view for matching, not a modification to the event record.
All estimates use observed associations and can themselves be affected by faults.
"""
from collections import Counter,defaultdict
import math

from .data import Event,Graph


def field_stability(graph,target_types,min_entities=8,min_observations=3):
    if min_entities<2 or min_observations<2:
        raise ValueError('Reliability needs multiple entities and repeated observations')
    values=defaultdict(lambda:defaultdict(Counter))
    for event in graph.events:
        for obj in event.participants:
            if graph.objects[obj] in target_types:
                for field,value in event.attributes.items():
                    values[graph.objects[obj],field][obj][value]+=1
    result={}
    for (typ,field),groups in sorted(values.items()):
        groups=[counts for counts in groups.values() if sum(counts.values())>=min_observations]
        n=len(groups);total=Counter();within=[];squared=[]
        for counts in groups:
            size=sum(counts.values())
            within.append(sum(v*(v-1) for v in counts.values())/(size*(size-1)))
            squared.append(sum((v/size)**2 for v in counts.values()))
            for value,count in counts.items():total[value]+=count/size
        # Equal entity weighting; between-entity agreement excludes self-pairs.
        between=((sum(value*value for value in total.values())-sum(squared))/(n*(n-1))) if n>1 else 1.
        observed=sum(within)/n if n else 0.
        score=max(0.,min(1.,(observed-between)/max(1e-12,1-between))) if n>=min_entities and between<1-1e-12 else 0.
        result.setdefault(typ,{})[field]={'entities':n,'within_entity_agreement':observed,
                                         'between_entity_agreement':max(0.,min(1.,between)),
                                         'adjusted_stability':score}
    return result


def stable_attribute_view(graph,target_types,threshold=.2,min_entities=8,min_observations=3):
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid stability threshold')
    profile=field_stability(graph,target_types,min_entities,min_observations)
    retained={field for fields in profile.values() for field,stats in fields.items()
              if stats['entities']>=min_entities and stats['adjusted_stability']>threshold}
    view=Graph(graph.name,dict(graph.objects),[
        Event(e.uid,e.time,e.kind,e.participants,{k:v for k,v in e.attributes.items() if k in retained})
        for e in graph.events])
    return view,{'threshold':threshold,'min_entities':min_entities,'min_observations':min_observations,
                 'retained_fields':sorted(retained),'profile':profile,
                 'limit':'Exact-value stability of observed entity histories, not proof of identity or field truth.'}
