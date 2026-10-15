from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tgrepair.data import Event, Graph, anonymize, save_graph, sha256
from tgrepair.corruption import mixed_faults, remove_participation
from tgrepair.matching import ObservedIndex, expected_topk_recall


def small_graph():
    return Graph('fixture', {'customer_secret_A': 'Case', 'customer_secret_B': 'Case', 'worker': 'Resource'},
                 [Event('source_1', 0, 'start', ('customer_secret_A', 'worker')),
                  Event('source_2', 1, 'start', ('customer_secret_B', 'worker')),
                  Event('source_3', 2, 'end', ('customer_secret_A', 'worker')),
                  Event('source_4', 3, 'end', ('customer_secret_B', 'worker'))])


def test_hidden_edges_are_absent_from_index():
    clean = anonymize(small_graph())
    observed, labels = remove_participation(clean, 7, 'contiguous', {'Case'}, rate=1.0)
    index = ObservedIndex(observed)
    assert len(labels) == 4
    for hole in labels:
        assert hole.reference_object not in index.hist
        assert all(hole.reference_object not in pairs for pairs in index.pairs.values())
        candidates = index.candidates(hole.event_uid, hole.object_type)
        f = index.features(hole.event_uid, candidates)
        assert np.all(f[:, :12] == 0)


def test_ids_and_source_order_are_not_observed():
    result = anonymize(small_graph())
    assert not any('secret' in o for o in result.objects)
    assert not any('source' in e.uid for e in result.events)
    assert set(result.objects.values()) == {'Case', 'Resource'}
    assert sum(len(e.participants) for e in result.events) == 8


def test_repeated_serialization_has_same_hash(tmp_path):
    graph = anonymize(small_graph())
    p, q = tmp_path / 'a.json.gz', tmp_path / 'b.json.gz'
    save_graph(graph, p)
    save_graph(graph, q)
    assert sha256(p) == sha256(q)


def test_multi_target_and_tied_ranking():
    assert expected_topk_recall(np.ones(4), np.array([True, True, False, False]), 2) == 0.5
    assert expected_topk_recall(np.array([2, 1, 0]), np.array([True, False, True]), 2) == 0.5
    assert expected_topk_recall(np.array([2, 1]), np.array([True, False]), 3) == 1 / 3


def test_faults_do_not_change_reference_execution():
    clean = anonymize(small_graph())
    before = [(e.uid, e.kind, e.time, e.participants) for e in clean.events]
    observed, labels = remove_participation(clean, 0, 'independent', {'Case'}, rate=1.0)
    assert before == [(e.uid, e.kind, e.time, e.participants) for e in clean.events]
    assert all(len(e.participants) == 1 for e in observed.events)
    assert len(labels) == 4


def test_current_wrong_link_does_not_supply_its_own_temporal_evidence():
    graph = Graph('one_event', {'a': 'Case', 'b': 'Case', 'r': 'Resource'},
                  [Event('e', 0, 'activity', ('a', 'r'))])
    index = ObservedIndex(graph)
    candidates = index.candidates('e', 'Case', allow_current=True)
    assert set(candidates) == {'a', 'b'}
    f = index.features('e', candidates, exclude_query=True)
    assert np.all(f[:, :12] == 0)


def test_fragmentation_handles_do_not_reveal_aliases_and_reference_replays():
    base = small_graph()
    graph = Graph('extended', base.objects,
                  base.events + [Event('source_5', 4, 'retry', ('customer_secret_A', 'worker')),
                                 Event('source_6', 5, 'end', ('customer_secret_A', 'worker'))])
    observed, reference = mixed_faults(graph, 9, {'Case'}, missing_rate=0, wrong_rate=0, fragment_rate=1)
    assert len(reference['fragmentations']) == 1
    assert all(o.startswith('o') and len(o) == 10 for o in observed.objects)
    identity = reference['identity']
    for e in observed.events:
        assert {identity[o] for o in e.participants} == set(reference['event_participation'][e.uid])


def test_same_time_activity_order_does_not_change_features():
    objects = {'a': 'Case', 'b': 'Case', 'r': 'Resource'}
    rows = [Event('a1', 0, 'start', ('a', 'r')), Event('a2', 0, 'check', ('a', 'r')),
            Event('query', 1, 'continue', ('r',)), Event('a3', 2, 'end', ('a', 'r'))]
    first = ObservedIndex(Graph('one', objects, rows))
    second = ObservedIndex(Graph('two', objects, [rows[1], rows[0], *rows[2:]]))
    np.testing.assert_allclose(first.features('query', ['a', 'b']), second.features('query', ['a', 'b']))


def test_legitimate_attributes_survive_faults_without_self_confirmation():
    g = Graph('attrs', {'a':'Case','b':'Case','r':'Resource'},
              [Event('e0',0,'start',('a','r'),{'amount':5}),
               Event('e1',1,'start',('b','r'),{'amount':10}),
               Event('e2',2,'end',('b','r'),{'amount':5})])
    f = ObservedIndex(g).features('e2',['a','b'],exclude_query=True)
    assert f[0,13] == 1
    assert f[1,13] == 0
    damaged,_ = remove_participation(g,0,'independent',{'Case'},rate=1)
    assert [e.attributes for e in damaged.events] == [e.attributes for e in g.events]
