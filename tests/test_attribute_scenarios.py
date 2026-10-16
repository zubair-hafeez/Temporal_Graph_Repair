from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.attribute_scenarios import generate_attribute_view,PROFILES
from tgrepair.data import Event,Graph


def test_observation_scenarios_are_deterministic_and_never_copy_entity_handles():
    graph=Graph('fixture',{'SECRET-A':'Case'},[
        Event(f'e{i}',float(i),['Start','Work','Finish'][i%3],('SECRET-A',),{'Amount':100.}) for i in range(12)])
    before=[dict(e.attributes) for e in graph.events]
    for profile in PROFILES:
        first,recipe=generate_attribute_view(graph,19,profile)
        second,_=generate_attribute_view(graph,19,profile)
        assert first==second
        assert 'SECRET' not in repr(first)
        assert set(first)=={e.uid for e in graph.events}
        assert [e.attributes for e in graph.events]==before
        if profile=='original':assert list(first.values())==before
