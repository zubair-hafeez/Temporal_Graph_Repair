"""Audit the complete frozen coupled suite from saved observations and repairs.

Primary history F1, exact recovery and expanded damage are recomputed here
without calling the implementation that produced those reported metrics.
Edit logs are replayed without rescoring. This does not establish independent
model retraining or independent expert verification of the source records.
"""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.confirmation import verify_protocol
from revision.audit_native_results import require_close
from tgrepair.data import load_graph, sha256
from tgrepair.edits import Edit, RepairState
from tgrepair.application_queries import audit_queries
from tgrepair.metrics import identity_pair_scores


def independent_metrics(before, after, reference, types, clusters):
    truth, aliases, found = defaultdict(set), defaultdict(list), defaultdict(set)
    for uid, objects in reference['event_participation'].items():
        for obj in objects:
            if before.objects[obj] in types:
                truth[obj].add(uid)
    for handle, identity in reference['identity'].items():
        if before.objects[handle] in types:
            aliases[identity].append(handle)
    cluster_identities = defaultdict(set)
    for handle, representative in clusters.items():
        if before.objects[handle] in types:
            cluster_identities[representative].add(reference['identity'][handle])
    for event in after.events:
        for obj in event.participants:
            if after.objects[obj] in types:
                found[clusters[obj]].add(event.uid)
    f1, exact = [], []
    for identity, expected in truth.items():
        values, matches = [], []
        for handle in aliases[identity]:
            retrieved = found[clusters[handle]]
            values.append(2 * len(expected.intersection(retrieved)) / (len(expected) + len(retrieved)))
            matches.append(float(expected == retrieved))
        f1.append(statistics.mean(values)); exact.append(statistics.mean(matches))
    correct_before = damaged = repaired = 0
    old_events = {event.uid: event for event in before.events}
    if len(old_events) != len(after.events) or set(old_events) != {e.uid for e in after.events}:
        raise AssertionError('Event set changed')
    for event in after.events:
        old = old_events[event.uid]
        if (event.time, event.kind, event.attributes) != (old.time, old.kind, old.attributes):
            raise AssertionError('Recorded event content changed')
        old_context = {o for o in old.participants if before.objects[o] not in types}
        new_context = {o for o in event.participants if after.objects[o] not in types}
        if old_context != new_context:
            raise AssertionError('Context relations changed')
        expected = {o for o in reference['event_participation'][event.uid] if before.objects[o] in types}
        prior = {reference['identity'][o] for o in old.participants if before.objects[o] in types}
        obtained = set()
        for obj in event.participants:
            if after.objects[obj] in types:
                obtained.update(cluster_identities[clusters[obj]])
        correct_before += int(prior == expected)
        damaged += int(prior == expected and obtained != expected)
        repaired += int(prior != expected and obtained == expected)
    return {'f1': statistics.mean(f1), 'exact_history_fraction': statistics.mean(exact),
            'entities': len(truth), 'originally_correct_events': correct_before,
            'originally_incorrect_events': len(before.events) - correct_before,
            'expanded_new_damage': damaged, 'expanded_repaired_events': repaired}


def audit():
    protocol = verify_protocol()
    protocol_hash = sha256(ROOT / 'revision/docs/COUPLED_CONFIRMATION_PROTOCOL_V1.json')
    reports = ROOT / 'revision/reports/coupled_confirmation_v1'
    expected_names = {j['id'] + '.json' for j in protocol['jobs']}
    if {p.name for p in reports.glob('*.json')} != expected_names:
        raise RuntimeError('Complete every declared job before the final audit')
    result = {'status': 'auditing_coupled_confirmation', 'protocol_sha256': protocol_hash,
              'audit_script_sha256': sha256(__file__), 'jobs': [], 'inputs': {}}
    tasks = {}
    for job in protocol['jobs']:
        path = reports / (job['id'] + '.json')
        report = json.loads(path.read_text())
        if report['status'] != 'completed_coupled_confirmation' or report['pins']['job'] != job:
            raise AssertionError('Incomplete or changed job: ' + job['id'])
        if report['pins']['protocol_sha256'] != protocol_hash:
            raise AssertionError('Protocol mismatch')
        key = (job['domain'], job['seed'], job['variant'])
        directory = ROOT / 'revision/data/confirmation_tasks_v1' / job['domain'] / f'{job["variant"]}_seed{job["seed"]}'
        task_path = directory / 'task.json'
        task = json.loads(task_path.read_text())
        if sha256(task_path) != report['pins']['task_sha256'] or task['pins']['protocol_sha256'] != protocol_hash:
            raise AssertionError('Task identity mismatch')
        for name, digest in task['files'].items():
            if sha256(directory / name) != digest:
                raise AssertionError('Task content changed')
        if key not in tasks:
            tasks[key] = (load_graph(directory / 'observed.json.gz'), json.loads((directory / 'reference.json').read_text()))
        before, reference = tasks[key]
        output = ROOT / 'revision/data/coupled_decoding_v1' / job['id']
        for name, digest in report['artifacts'].items():
            if sha256(output / name) != digest:
                raise AssertionError('Prediction artifact changed')
        after = load_graph(output / 'repaired.json.gz')
        clusters = json.loads((output / 'clusters.json').read_text())
        edits = json.loads((output / 'audit.json').read_text())
        if before.objects != after.objects or set(clusters) != set(before.objects):
            raise AssertionError('Object catalog changed')
        for handle, representative in clusters.items():
            if clusters[representative] != representative or before.objects[handle] != before.objects[representative]:
                raise AssertionError('Invalid or cross-type cluster')
        state = RepairState(before)
        for batch in edits['rounds']:
            state.apply_batch([Edit(**entry['edit']) for entry in batch['edits']])
        if state.graph.events != after.events or state.clusters != clusters:
            raise AssertionError('Saved edit replay differs')
        if sum(len(batch['edits']) for batch in edits['rounds']) != report['edits']:
            raise AssertionError('Edit count differs')
        types = set(protocol['domains'][job['domain']]['target_types'])
        metrics = independent_metrics(before, after, reference, types, clusters)
        for field, value in metrics.items():
            expected = report['history'][field] if field in {'f1', 'exact_history_fraction', 'entities'} else report['events'][field]
            require_close(value, expected, job['id'] + '.' + field)
        require_close(audit_queries(after, reference, types, clusters), report['queries'], 'queries')
        require_close(identity_pair_scores(after, reference, types, clusters), report['identity'], 'identity')
        result['inputs'][path.relative_to(ROOT).as_posix()] = sha256(path)
        result['jobs'].append({'id': job['id'], **metrics, 'edit_replay_and_content_preservation': True})
        print(json.dumps({'audited': job['id']}), flush=True)
    result['status'] = 'all_coupled_confirmation_predictions_and_edit_replays_verified'
    destination = ROOT / 'revision/reports/coupled_confirmation_audit_v1.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    audit()
