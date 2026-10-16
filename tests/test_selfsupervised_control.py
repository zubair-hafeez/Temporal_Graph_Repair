from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from tgrepair.data import Event,Graph
from prepare_selfsupervised_control import temporal_pseudo_tasks
from prepare_source_edit_controls import build


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


def test_training_labels_cover_both_target_types_without_cross_type_edits():
    objects={'a':'Order','b':'Item','other_a':'Order','other_b':'Item','r':'Resource'}
    graph=Graph('typed',objects,[
        Event('e0',0.,'Work',('a','b','r')),
        Event('e1',1.,'Work',('r',)),
        Event('e2',2.,'Work',('other_a','other_b','r'))])
    reference={'identity':{o:o for o in objects},'event_participation':{
        'e0':['a','b','r'],'e1':['a','b','r'],'e2':['other_a','other_b','r']}}
    _,labels,operations,counts=build(graph,reference,None,100,{'Order','Item'})
    assert labels[operations==0].sum()==2
    assert labels[operations==1].sum()==0
    assert labels[operations==2].sum()==0
    assert counts['candidate_pairs']==2
