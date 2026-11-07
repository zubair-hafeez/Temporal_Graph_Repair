from types import SimpleNamespace
import numpy as np
from tgrepair.data import Graph, Event
from tgrepair.edits import Edit
from .method import apply_policy


def test_never_merge_keeps_atomic_recovery_and_rejects_confident_merge():
    graph = Graph('test', {'a':'Root', 'b':'Root', 'c':'Item', 'd':'Item'},
                  [Event('e1', 1., 'open', ('a','c')), Event('e2', 2., 'close', ('d',))])
    obs = SimpleNamespace(graph=graph, evidence=graph, edits=[
        Edit('MERGE_ALIASES','c','d'), Edit('REMOVE_PARTICIPANT','e1','a'),
        Edit('ADD_PARTICIPANT','e1','b')], operations=np.array([2,1,0]))
    scores = np.array([1.,.999,.999]); thresholds = [.9]*3
    guarded, audit = apply_policy(obs,scores,thresholds,'Root','guarded')
    assert guarded.clusters['c'] == guarded.clusters['d']
    never, audit = apply_policy(obs,scores,thresholds,'Root','never_merge')
    assert len(audit['edits']) == 2
    assert never.graph.events[0].participants == ('b','c')
    assert thresholds == [.9]*3
    never.undo(); assert never.graph == graph


def test_never_merge_retains_tie_abstention():
    graph = Graph('test', {'a':'Root','b':'Root'}, [Event('e',0.,'record',())])
    obs = SimpleNamespace(graph=graph,evidence=graph,edits=[Edit('ADD_PARTICIPANT','e','a'),
        Edit('ADD_PARTICIPANT','e','b')],operations=np.array([0,0]))
    state,audit=apply_policy(obs,np.array([.999,.999]),[.9]*3,'Root','never_merge')
    assert not audit['edits'] and state.graph == graph
