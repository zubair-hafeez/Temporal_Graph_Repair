import copy
import numpy as np

from revision.annotation import allocate, make_verified_view, build_examples
from tgrepair.data import Graph, Event


def test_nested_budgets_include_calibration_and_keep_roles():
    assignments = allocate([str(i) for i in range(300)], 5)
    for budget, parts in assignments.items():
        assert not parts['fit'] & parts['calibration']
        assert len(parts['fit'] | parts['calibration']) == budget
        assert len(parts['calibration']) == budget // 4
    for role in ['fit', 'calibration']:
        assert assignments[16][role] < assignments[64][role] < assignments[256][role]


def test_unverified_reference_changes_cannot_change_training_examples():
    graph = Graph('toy', {'a': 'Case', 'alias': 'Case', 'b': 'Case', 'c': 'Case', 'r': 'Resource'}, [
        Event('e0', 0., 'start', ('a', 'r')),
        Event('e1', 1., 'finish', ('b', 'r')),
        Event('e2', 2., 'start', ('b', 'r')),
        Event('e3', 3., 'finish', ('c', 'r')),
        Event('e4', 4., 'finish', ('alias', 'r')),
    ])
    reference = {'identity': {'a': 'a', 'alias': 'a', 'b': 'b', 'c': 'c', 'r': 'r'},
                 'event_participation': {'e0': ['a', 'r'], 'e1': ['a', 'r'], 'e2': ['b', 'r'],
                                         'e3': ['c', 'r'], 'e4': ['a', 'r']}}
    changed = copy.deepcopy(reference)
    changed['identity']['c'] = 'b'
    changed['event_participation']['e2'] = ['c', 'r']
    changed['event_participation']['e3'] = ['b', 'r']
    left = make_verified_view(graph, reference, {'a': ['a']}, {'a'}, {'Case'})
    right = make_verified_view(graph, changed, {'a': ['a']}, {'a'}, {'Case'})
    assert left == right
    a = build_examples(graph, left, {'Case'})
    b = build_examples(graph, right, {'Case'})
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(x, y)
    assert a[3] == b[3]
    assert set(left.alias_owner) == {'a', 'alias'}
