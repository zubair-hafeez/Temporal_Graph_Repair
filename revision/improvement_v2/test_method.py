from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from revision.annotation import allocate
from tgrepair.data import Event, Graph
from tgrepair.edits import Edit
from revision.improvement_v2.method import (actions, ambiguous_graph, corrupt, decode,
                                          project, training_labels, workflow)
from revision.improvement_v2.run import subset


def test_projection_removes_retrieval_information():
    graph, _ = workflow(15, roots=4)
    different_times = Graph(graph.name, graph.objects,
                            [replace(e, time=e.time * 31 + 17) for e in graph.events])
    a = actions(graph, {'Root', 'Item'}, 'no_time')
    b = actions(different_times, {'Root', 'Item'}, 'no_time')
    assert a.edits == b.edits
    np.testing.assert_array_equal(a.action_features, b.action_features)
    different_fields = Graph(graph.name, graph.objects,
                             [replace(e, attributes={'hidden': e.uid}) for e in graph.events])
    a = actions(graph, {'Root', 'Item'}, 'no_attributes')
    b = actions(different_fields, {'Root', 'Item'}, 'no_attributes')
    assert a.edits == b.edits
    np.testing.assert_array_equal(a.action_features, b.action_features)
    view = project(graph, {'Root', 'Item'}, 'no_context')
    assert all(view.objects[o] in {'Root', 'Item'} for e in view.events for o in e.participants)
    assert all(not e.participants for e in project(graph, {'Root', 'Item'}, 'no_history').events)


def test_ambiguous_additions_abstain_even_at_confident_scores():
    obs = actions(ambiguous_graph(3, pairs=2), {'Root'})
    scores = np.full(len(obs.edits), .999)
    state, audit = decode(obs, scores, [.9, .9, .9], 'Root', safe=True)
    assert not audit['edits']
    assert all(not e.participants for e in state.graph.events)
    unsafe, audit = decode(obs, scores, [.9, .9, .9], 'Root', safe=False)
    assert len(audit['edits']) == 2


def test_merge_guard_and_full_undo():
    graph = Graph('cooccurrence', {'a': 'Root', 'b': 'Root'},
                  [Event('e', 1., 'record', ('a', 'b'), {'value': 2})])
    obs = SimpleNamespace(graph=graph, evidence=graph, edits=[Edit('MERGE_ALIASES', 'a', 'b')],
                          operations=np.array([2]))
    state, audit = decode(obs, np.array([.999]), [.9, .9, .9], 'Root', safe=True)
    assert not audit['edits']
    state, audit = decode(obs, np.array([.999]), [.9, .9, .9], 'Root', safe=False)
    assert state.clusters['b'] == 'a'
    assert state.graph.events[0].attributes == {'value': 2}
    state.undo()
    assert state.graph.events == graph.events
    assert state.clusters == {'a': 'a', 'b': 'b'}


def test_atomic_root_replacement_and_no_unpaired_deletion():
    graph = Graph('root', {'a': 'Root', 'b': 'Root'}, [Event('e', 0., 'record', ('a',))])
    obs = SimpleNamespace(graph=graph, evidence=graph,
                          edits=[Edit('REMOVE_PARTICIPANT', 'e', 'a'), Edit('ADD_PARTICIPANT', 'e', 'b')],
                          operations=np.array([1, 0]))
    state, _ = decode(obs, np.array([.99, .1]), [.9] * 3, 'Root')
    assert state.graph.events == graph.events
    state, audit = decode(obs, np.array([.99, .99]), [.9] * 3, 'Root')
    assert state.graph.events[0].participants == ('b',)
    assert len(audit['edits']) == 2
    state.undo(); assert state.graph.events == graph.events


def test_supported_merge_requires_mutual_margin_and_preserves_events():
    graph = Graph('separate', {'a': 'Root', 'b': 'Root', 'c': 'Root'},
                  [Event('e1', 1., 'open', ('a',), {'x': 3}),
                   Event('e2', 2., 'cancel', ('b',), {'x': 5}),
                   Event('e3', 3., 'retry', ('c',), {'x': 7})])
    obs = SimpleNamespace(graph=graph, evidence=graph,
                          edits=[Edit('MERGE_ALIASES', 'a', 'b')], operations=np.array([2]))
    state, audit = decode(obs, np.array([.999]), [.9] * 3, 'Root')
    assert len(audit['edits']) == 1 and state.clusters['a'] == state.clusters['b']
    for original, repaired in zip(graph.events, state.graph.events):
        assert (original.uid, original.time, original.kind, original.attributes) == (repaired.uid, repaired.time, repaired.kind, repaired.attributes)
    state.undo(); assert state.graph == graph
    obs.edits += [Edit('MERGE_ALIASES', 'a', 'c'), Edit('MERGE_ALIASES', 'b', 'c')]
    obs.operations = np.array([2, 2, 2])
    state, audit = decode(obs, np.array([.999, .998, .4]), [.9] * 3, 'Root')
    assert not audit['edits']


def test_corruptions_preserve_execution_content_and_labels():
    clean, _ = workflow(31, roots=8)
    by_id = {e.uid: e for e in clean.events}
    for variant in ['mixed', 'independent', 'mild', 'severe', 'missing', 'wrong', 'fragment', 'pipeline', 'clean']:
        graph, reference = corrupt(clean, {'Root', 'Item'}, 99, variant)
        for e in graph.events:
            prior = by_id[reference['event_origin'][e.uid]]
            assert (e.time, e.kind, e.attributes) == (prior.time, prior.kind, prior.attributes)
        obs = actions(graph, {'Root', 'Item'}, event_limit=8)
        labels = training_labels(obs, reference)
        assert len(labels) == len(obs.edits)
        if variant == 'clean': assert labels.sum() == 0


def test_nested_verified_history_budget_and_bundle_boundary():
    roots = [f'r{i}' for i in range(300)]
    allocations = allocate(roots, 31)
    for budget, split in allocations.items():
        assert len(split['fit']) + len(split['calibration']) == budget
        assert not split['fit'] & split['calibration']
    assert allocations[16]['fit'] <= allocations[64]['fit'] <= allocations[256]['fit']
    assert allocations[16]['calibration'] <= allocations[64]['calibration'] <= allocations[256]['calibration']
    graph = Graph('two', {'r0': 'Root', 'r1': 'Root', 'c': 'Resource'},
                  [Event('e0', 0., 'x', ('r0', 'c')), Event('e1', 1., 'x', ('r1', 'c'))])
    selected = subset(graph, {'r0': ['r0'], 'r1': ['r1']}, {'r0'})
    assert [e.uid for e in selected.events] == ['e0']
    assert 'r1' not in selected.objects


def test_generator_has_retries_cancellations_and_is_reproducible():
    a, count = workflow(32)
    b, second = workflow(32)
    assert a == b and count == second
    assert count.get('cancel', 0) > 0
    assert sum(n for k, n in count.items() if k.startswith('retry_')) > 0
    a.validate()
