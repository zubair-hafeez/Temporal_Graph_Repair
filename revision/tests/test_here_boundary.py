from revision.here_benchmark import prepare_records, project_single_case, sequence, tokenizer, make_dataset, author_greedy_decoder
from tgrepair.data import Graph, Event


def test_native_projection_and_hidden_identifier_boundary():
    graph = Graph('toy', {'a': 'Case', 'b': 'Case', 'child': 'Item', 'r': 'Resource'}, [
        Event(str(i), float(i), 'activity', ('a' if i % 2 else 'b', 'child', 'r'), {'amount': i})
        for i in range(80)])
    projected = project_single_case(graph, 'Case', 2)
    assert 'child' not in projected.objects
    assert all(not e.attributes for e in projected.events)
    rows, hidden, truth, number_case, observed, reference = prepare_records(projected, 'Case', .3, 0)
    assert hidden and all(r['label'] == '[NONE]' for r in rows if r['uid'] in hidden)
    assert all(not any(observed.objects[o] == 'Case' for o in e.participants)
               for e in observed.events if e.uid in hidden)
    known = [r for r in rows if r['uid'] not in hidden]
    windows = [sequence(known[i:i + 15]) for i in range(len(known) - 14)]
    assert all('[NONE]' not in window['target'] for window in windows)
    src, tgt = tokenizer(windows, 'discrete'), tokenizer(windows, 'target')
    item = make_dataset(windows, src, tgt)[0]
    assert item['encoder_input'].shape == (32,)
    assert item['decoder_input'].shape == (17,)
    assert item['continuous_input'].shape == (17,)
    assert callable(author_greedy_decoder())
