"""Verify every declared native result against saved predictions and source IDs.

The accuracy check constructs its reference directly from the pinned source
graph, independently of the masking/preparation helper used by the runners.
Run only after all six native suites' constituent jobs have completed.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.native_protocol import verify_protocol
from tgrepair.data import Graph, load_graph, sha256
from tgrepair.metrics import traceability


def require_close(actual, reported, path='metric'):
    if isinstance(actual, dict):
        if set(actual) != set(reported):
            raise AssertionError('Metric keys differ: ' + path)
        for key in actual:
            require_close(actual[key], reported[key], path + '.' + key)
    elif isinstance(actual, (float, int)):
        if abs(actual - reported) > 1e-12:
            raise AssertionError('Metric value differs: ' + path)
    elif actual != reported:
        raise AssertionError('Metric value differs: ' + path)


def audit():
    protocol = verify_protocol('bpi2017', 0, .3, 30, 2048)
    protocol_path = ROOT / 'revision/docs/NATIVE_CONFIRMATION_PROTOCOL_V1.json'
    result = {'status': 'auditing_native_confirmation', 'protocol_sha256': sha256(protocol_path),
              'audit_script_sha256': sha256(__file__), 'jobs': [], 'inputs': {}}
    for domain, seed, rate, epochs, roots in protocol['jobs']:
        base = f'{domain}_seed{seed}_missing{int(rate * 100)}_roots{roots}'
        names = {'here': f'here_native_{base}_epochs{epochs}_v1',
                 'controls': f'native_controls_confirmation_{base}_v1'}
        reports = {}
        for key, name in names.items():
            path = ROOT / 'revision/reports' / (name + '.json')
            reports[key] = json.loads(path.read_text())
            expected_status = 'completed_here_native_confirmation' if key == 'here' else 'completed_native_controls_confirmation'
            if reports[key]['status'] != expected_status:
                raise RuntimeError('Native job remains incomplete: ' + name)
            if reports[key]['protocol_sha256'] != result['protocol_sha256']:
                raise ValueError('Result belongs to a different protocol')
            result['inputs'][str(path.relative_to(ROOT))] = sha256(path)
        metadata = json.loads((ROOT / 'revision/reports' / f'native_cohort_{domain}_v1.json').read_text())
        root_type = metadata['root_type']
        source = ROOT / 'revision/data/native_cohort_v1' / domain / 'reference_graph.json.gz'
        clean = load_graph(source)
        truth = {}
        for event in clean.events:
            cases = [o for o in event.participants if clean.objects[o] == root_type]
            if len(cases) != 1:
                raise AssertionError('Native source violates single-case contract')
            truth[event.uid] = cases[0]
        here_dir = ROOT / 'revision/checkpoints' / names['here']
        controls_dir = ROOT / 'revision/data/native_controls' / names['controls']
        observed_path = here_dir / 'observed.json.gz'
        observed_hash = sha256(observed_path)
        if observed_hash != sha256(controls_dir / 'observed.json.gz'):
            raise AssertionError('Methods received different observations')
        for key in reports:
            if reports[key]['observed_graph_sha256'] != observed_hash:
                raise AssertionError('Observed input checksum mismatch')
        observed = load_graph(observed_path)
        if {e.uid for e in observed.events} != set(truth):
            raise AssertionError('Native evaluation omitted source events')
        hidden = set()
        for event in observed.events:
            cases = [o for o in event.participants if observed.objects[o] == root_type]
            if not cases:
                hidden.add(event.uid)
            elif cases != [truth[event.uid]]:
                raise AssertionError('Unmasked IDs were not preserved in the input')
        reference = {'identity': {o: o for o in observed.objects},
                     'event_participation': {uid: [case] for uid, case in truth.items()}}
        here_predictions_path = here_dir / 'predictions.json'
        controls_predictions_path = controls_dir / 'predictions.json'
        for path, key in [(here_predictions_path, 'here'), (controls_predictions_path, 'controls')]:
            if sha256(path) != reports[key]['predictions_sha256']:
                raise AssertionError('Prediction checksum mismatch')
        if sha256(here_dir / 'model.pt') != reports['here']['model_sha256']:
            raise AssertionError('HERE checkpoint changed')
        if sha256(controls_dir / 'known_labels_only.joblib') != reports['controls']['known_labels_model_sha256']:
            raise AssertionError('Known-label control checkpoint changed')
        predictions = json.loads(controls_predictions_path.read_text())
        predictions['here'] = json.loads(here_predictions_path.read_text())['predicted']
        job = {'domain': domain, 'seed': seed, 'events': len(truth), 'hidden': len(hidden),
               'identical_observed_inputs': True, 'methods': {}}
        for method, predicted in predictions.items():
            if not set(predicted) <= hidden:
                raise AssertionError('Prediction modifies an unmasked event')
            if any(o is not None and observed.objects.get(o) != root_type for o in predicted.values()):
                raise AssertionError('Prediction leaves the typed available catalog')
            correct = sum(predicted.get(uid) == truth[uid] for uid in hidden)
            assigned = sum(predicted.get(uid) is not None for uid in hidden)
            score = correct / max(1, len(hidden))
            expected = reports['here']['heldout_missing_id'] if method == 'here' else reports['controls']['results'][method]
            for key, value in [('correct', correct), ('assigned', assigned), ('hidden', len(hidden))]:
                if expected[key] != value:
                    raise AssertionError('Prediction count differs: ' + method + '.' + key)
            require_close(score, expected['accuracy_all_missing'], method + '.accuracy')
            events = [replace(e, participants=tuple(sorted(set(e.participants) |
                      ({predicted[e.uid]} if predicted.get(e.uid) is not None else set())))) for e in observed.events]
            repaired = Graph(observed.name + '_audit', dict(observed.objects), events)
            history = traceability(repaired, reference, {root_type})
            require_close(history, reports['here']['history_after'] if method == 'here' else expected['history'], method + '.history')
            if method == 'here':
                saved = load_graph(here_dir / 'repaired.json.gz')
                if saved.events != repaired.events or saved.objects != repaired.objects:
                    raise AssertionError('Saved HERE repaired graph differs from predictions')
            job['methods'][method] = {'correct': correct, 'assigned': assigned, 'accuracy': score,
                                       'source_reference_accuracy_and_history_verified': True}
        result['jobs'].append(job)
    result['status'] = 'all_native_confirmation_predictions_verified'
    destination = ROOT / 'revision/reports/native_confirmation_audit_v1.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'jobs': len(result['jobs']), 'report': str(destination)}))


if __name__ == '__main__':
    audit()
