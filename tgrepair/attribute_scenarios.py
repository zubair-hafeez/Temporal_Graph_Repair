"""Executable observation-field variation for generated workflow histories.

These scenarios change simulated measurements/descriptions, not execution or
recording-error labels. They never derive fields from target object identifiers.
"""
import random


PROFILES=('original','activity_context','measurements','mixed_fields')


def generate_attribute_view(graph,seed,profile):
    if profile not in PROFILES:raise ValueError(profile)
    rng=random.Random(seed)
    original_fields=sorted({k for event in graph.events for k in event.attributes})
    if profile=='original':
        return {e.uid:dict(e.attributes) for e in graph.events},{
            'profile':profile,'seed':seed,'retained_original_fields':original_fields,'additional_fields':[]}
    retained=[field for field in original_fields if rng.random() <
              (1. if profile=='activity_context' else .5)]
    count=rng.choice([2,4]) if profile=='activity_context' else rng.choice([2,4,6]) if profile=='measurements' else rng.choice([3,6,9,12])
    kinds=sorted({event.kind for event in graph.events})
    specs=[]
    for i in range(count):
        family=('activity_category' if profile=='activity_context' else 'measurement' if profile=='measurements'
                else rng.choice(['activity_category','measurement','incidental_category','time_band']))
        palette=rng.choice([2,4,8])
        spec={'name':f'ObservedField{i:02d}','family':family,'availability':rng.choice([.5,.8,1.]),'palette':palette}
        if spec['name'] in original_fields:raise ValueError('Generated observation field collision')
        if family=='activity_category':spec['kind_values']={k:rng.randrange(palette) for k in kinds}
        if family=='measurement':
            spec['kind_scales']={k:rng.choice([.1,.5,1.,2.,5.]) for k in kinds}
            spec['relative_noise']=rng.choice([.05,.2,1.])
        specs.append(spec)
    start=graph.events[0].time if graph.events else 0.
    span=max(1.,graph.events[-1].time-start) if graph.events else 1.
    result={}
    for event in graph.events:
        attrs={k:v for k,v in event.attributes.items() if k in retained}
        for spec in specs:
            if rng.random()>spec['availability']:continue
            family=spec['family']
            if family=='activity_category':value='category_'+str(spec['kind_values'][event.kind])
            elif family=='incidental_category':value='category_'+str(rng.randrange(spec['palette']))
            elif family=='time_band':value='band_'+str(min(spec['palette']-1,int((event.time-start)/span*spec['palette'])))
            else:
                base=event.attributes.get('Amount',100.)
                if not isinstance(base,(int,float)):base=100.
                value=round(abs(base)*spec['kind_scales'][event.kind]*max(.01,1+rng.gauss(0,spec['relative_noise'])),2)
            attrs[spec['name']]=value
        result[event.uid]=attrs
    return result,{'profile':profile,'seed':seed,'retained_original_fields':retained,'additional_fields':specs,
                   'limits':'Synthetic observation conditions; not measurements calibrated to an external organization.'}
