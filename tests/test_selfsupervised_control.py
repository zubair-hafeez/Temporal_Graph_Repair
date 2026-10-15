from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from tgrepair.data import Event,Graph
from prepare_selfsupervised_control import temporal_pseudo_tasks


def test_pseudo_reference_preserves_existing_wrong_observation():
    # 'wrong' is deliberately attached throughout. No clean identity or event
    # membership is supplied; extra-corruption training must not correct it.
    graph=Graph('imperfect',{'wrong':'Application','r':'Resource'},[
        Event(f'e{i}',float(i//2),'Work',('wrong','r')) for i in range(20)])
    tasks,boundary=temporal_pseudo_tasks(graph,{'Application'})
    train_times={e.time for e in tasks['train'][0].events}
    validation_times={e.time for e in tasks['validation'][0].events}
    assert not train_times&validation_times
    assert max(train_times)<boundary<=min(validation_times)
    for observed,reference in tasks.values():
        targets={o for ids in reference['event_participation'].values() for o in ids
                 if observed.objects[o]=='Application'}
        assert len(targets)==1
        assert all(sum(observed.objects[o]=='Application' for o in ids)==1
                   for ids in reference['event_participation'].values())
