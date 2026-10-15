from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tgrepair.data import Graph, Event
from tgrepair.metrics import identity_pair_scores, traceability
from tgrepair.identity_matching import match_aliases


def test_alias_merge_restores_history_but_overmerge_is_penalized():
    graph = Graph('fixture',{'a':'Case','alias':'Case','b':'Case'},
                  [Event('e1',0,'start',('a',)),Event('e2',1,'end',('alias',)),Event('e3',2,'start',('b',))])
    ref = {'identity':{'a':'a','alias':'a','b':'b'},
           'event_participation':{'e1':['a'],'e2':['a'],'e3':['b']}}
    correct = {'a':'a','alias':'a','b':'b'}
    wrong = {'a':'a','alias':'a','b':'a'}
    assert traceability(graph,ref,{'Case'},correct)['f1'] == 1
    assert traceability(graph,ref,{'Case'})['f1'] < 1
    assert traceability(graph,ref,{'Case'},wrong)['precision'] < 1
    assert identity_pair_scores(graph,ref,{'Case'},correct)['f1'] == 1
    assert identity_pair_scores(graph,ref,{'Case'},wrong)['precision'] == 1/3


def test_global_assignment_resolves_competing_alias_edges():
    graph = Graph('fixture',{o:'Case' for o in 'abcd'},[])
    edges = [('a','c',0.9),('a','d',0.85),('b','c',0.8)]
    result = match_aliases(graph,edges,{'Case'},threshold=0.5,global_assignment=True)
    assert result['a']==result['d']
    assert result['b']==result['c']
    assert result['a']!=result['b']
