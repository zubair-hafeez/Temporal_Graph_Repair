from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tgrepair.external_ocel import adapt_ocel


def test_adapter_omits_identity_copies_and_keeps_native_observations():
    raw = {'objects': [
        {'id': 'SECRET-A', 'type': 'Case', 'attributes': [{'name': 'key', 'value': 'SECRET-A'}],
         'relationships': [{'objectId': 'SECRET-B'}]},
        {'id': 'SECRET-B', 'type': 'Case'}, {'id': 'UNSEEN', 'type': 'Case'}],
        'events': [{'id': 'EVENT-SECRET', 'time': '2024-01-01T00:00:00Z', 'type': 'Cancel',
                    'relationships': [{'objectId': 'SECRET-A', 'qualifier': 'primary'}],
                    'attributes': [{'name': 'source_record_id', 'value': 'SECRET-A'},
                                   {'name': 'amount', 'value': 12.5},
                                   {'name': 'reason', 'value': 'linked to SECRET-B'},
                                   {'name': 'is_exception', 'value': 1},
                                   {'name': 'actor', 'value': 'ACTOR-SECRET'}]}]}
    graph, audit = adapt_ocel(raw, 'test', 0, 1)
    assert graph.events[0].attributes == {'amount': 12.5, 'is_exception': 1}
    assert len(graph.objects) == 2
    assert set(graph.objects.values()) == {'Case', 'NativeActor'}
    assert 'SECRET' not in repr(graph)
    assert audit['identifier_values_dropped'] == {'reason': 1}
    assert audit['fields_dropped'] == {'source_record_id': 1}


def test_time_window_never_splits_simultaneous_events():
    raw = {'objects': [{'id': 'a', 'type': 'Case'}], 'events': [
        {'id': str(i), 'type': 'Work', 'time': f'2024-01-01T00:00:{t:02d}Z',
         'relationships': [{'objectId': 'a'}]} for i, t in enumerate([0, 0, 1, 1, 2, 2, 3, 3])]}
    graph, audit = adapt_ocel(raw, 'test', .125, .625)
    assert len(graph.events) == 4
    assert {e.time for e in graph.events} == {1704067200., 1704067201.}
    assert audit['window']['stop_exclusive_utc_seconds'] == 1704067202.


def test_target_identifier_cannot_reenter_as_actor_context():
    raw = {'objects': [{'id': 'a', 'type': 'Case'}], 'events': [
        {'id': 'e', 'type': 'Work', 'time': '2024-01-01T00:00:00Z',
         'relationships': [{'objectId': 'a'}], 'attributes': [{'name': 'actor', 'value': 'a'}]}]}
    graph, audit = adapt_ocel(raw, 'test', 0, 1)
    assert len(graph.objects) == 1
    assert audit['identifier_values_dropped'] == {'actor': 1}
