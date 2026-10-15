from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.data import Event,Graph
from tgrepair.matching import ObservedIndex


def test_attribute_cache_matches_direct_history_with_query_exclusion():
    events=[Event('e0',0.,'A',('a',),{'Amount':10.,'Category':'x'}),
            Event('e1',1.,'B',('a','b'),{'Amount':20.,'Category':'y'}),
            Event('e2',2.,'C',('a',),{'Amount':20.}),
            Event('e3',3.,'D',('b',),{}),
            Event('e4',4.,'E',('a','b'),{'Amount':30.,'Category':'x'})]
    graph=Graph('fixture',{'a':'Case','b':'Case','cold':'Case'},events);index=ObservedIndex(graph)
    for event in events:
        if not event.attributes:continue
        for obj in graph.objects:
            for exclude in [False,True]:
                contains=exclude and obj in event.participants
                attrs=[e.attributes for e in events if obj in e.participants and not (exclude and e.uid==event.uid)]
                agree,conflict,coverage,num,numeric_count=0.,0.,0.,0.,0
                for key,value in event.attributes.items():
                    vals=[a[key] for a in attrs if key in a]
                    if vals:
                        match=sum(v==value for v in vals)/len(vals)
                        agree+=match;conflict+=1-match;coverage+=1
                        if isinstance(value,(int,float)) and all(isinstance(v,(int,float)) for v in vals):
                            num+=1/(1+abs(value-float(np.median(vals)))/index.attribute_scales[key]);numeric_count+=1
                exact=sum(all(a.get(k)==v for k,v in event.attributes.items()) for a in attrs)/max(1,len(attrs))
                expected=[agree/len(event.attributes),conflict/len(event.attributes),coverage/len(event.attributes),exact,num/max(1,numeric_count)]
                for _ in range(2):np.testing.assert_allclose(index.attribute_features(obj,event.attributes,contains),expected,rtol=0,atol=1e-15)
