from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.data import Event,Graph
from tgrepair.attribute_views import stable_attribute_view


def test_repeated_entity_evidence_is_separated_from_event_phase_and_constants():
    graph=Graph('fixture',{f'o{i}':'Case' for i in range(10)},[])
    for j in range(4):
        for i in range(10):
            graph.events.append(Event(f'e{j}_{i}',float(j),'Work',(f'o{i}',),{
                'entity_property':i,'event_phase':j%2,'constant':'same','measurement':j*10+i}))
    before=[dict(e.attributes) for e in graph.events]
    view,audit=stable_attribute_view(graph,{'Case'})
    assert audit['retained_fields']==['entity_property']
    assert all(set(e.attributes)=={'entity_property'} for e in view.events)
    assert [e.attributes for e in graph.events]==before
    assert [(e.uid,e.time,e.kind,e.participants) for e in graph.events]==[
        (e.uid,e.time,e.kind,e.participants) for e in view.events]


def test_small_or_singleton_histories_do_not_claim_reliable_fields():
    graph=Graph('fixture',{'a':'Case','b':'Case'},[
        Event('e0',0.,'Work',('a',),{'amount':10}),
        Event('e1',1.,'Work',('b',),{'amount':20})])
    view,audit=stable_attribute_view(graph,{'Case'})
    assert audit['retained_fields']==[]
    assert all(not e.attributes for e in view.events)
