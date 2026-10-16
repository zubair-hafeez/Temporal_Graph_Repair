from types import SimpleNamespace

import numpy as np

from revision.transactional_decoding import candidate_transactions
from tgrepair.data import Event, Graph
from tgrepair.edits import Edit, RepairState


def observation(participants):
    graph = Graph('toy', {'a': 'Case', 'b': 'Case', 'item': 'Item'},
                  [Event('e', 0., 'work', participants, {'value': 2})])
    edits = [Edit('REMOVE_PARTICIPANT' if o in participants else 'ADD_PARTICIPANT', 'e', o)
             for o in ['a', 'b', 'item']]
    ops = np.array([int(e.operation == 'REMOVE_PARTICIPANT') for e in edits])
    return SimpleNamespace(graph=graph, edits=edits, operations=ops)


def test_root_replacement_requires_both_scores_and_commits_as_one_transaction():
    obs = observation(('a', 'item'))
    # Unsupported addition cannot authorize deletion of the required sole root.
    assert candidate_transactions(obs, np.array([.99, .2, .1]), [.95] * 3, 'Case', set(), set()) == []
    tx = candidate_transactions(obs, np.array([.99, .98, .1]), [.95] * 3, 'Case', set(), set())
    assert len(tx) == 1 and len(tx[0]) == 2
    state = RepairState(obs.graph)
    state.apply_batch([edit for edit, _ in tx[0]])
    assert set(state.graph.events[0].participants) == {'b', 'item'}
    assert state.graph.events[0].attributes == {'value': 2}
    state.undo()
    assert state.graph.events == obs.graph.events


def test_missing_root_gets_only_one_candidate_and_reversals_are_blocked():
    obs = observation(('item',))
    tx = candidate_transactions(obs, np.array([.96, .99, .1]), [.95] * 3, 'Case', set(), set())
    assert tx == [[(Edit('ADD_PARTICIPANT', 'e', 'b'), .99)]]
    tx = candidate_transactions(obs, np.array([.96, .99, .1]), [.95] * 3, 'Case', set(),
                                {('REMOVE_PARTICIPANT', 'e', 'b')})
    assert tx == [[(Edit('ADD_PARTICIPANT', 'e', 'a'), .96)]]
