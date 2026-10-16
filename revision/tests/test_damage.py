from revision.evaluation import damage_counts
from tgrepair.data import Graph, Event


def test_wrong_alias_merge_expands_damage_beyond_representative_assignment():
    original = Graph('toy', {'a': 'Case', 'b': 'Case'}, [
        Event('e0', 0., 'start', ('a',)), Event('e1', 1., 'finish', ('b',))])
    merged = Graph('merged', dict(original.objects), [
        Event('e0', 0., 'start', ('a',)), Event('e1', 1., 'finish', ('a',))])
    reference = {'identity': {'a': 'a', 'b': 'b'}, 'event_participation': {'e0': ['a'], 'e1': ['b']}}
    result = damage_counts(original, merged, reference, {'Case'}, {'a': 'a', 'b': 'a'})
    assert result['originally_correct_events'] == 2
    assert result['direct_new_damage'] == 1
    assert result['expanded_new_damage'] == 2
