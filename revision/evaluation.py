"""Revision evaluation; sealed partitions require a separate frozen protocol."""
from collections import defaultdict
import json
from pathlib import Path
import time

from threadpoolctl import threadpool_limits

from tgrepair.corruption import mixed_faults
from tgrepair.data import load_graph, save_graph, sha256
from tgrepair.decoding import FeatureScorer, decode
from tgrepair.metrics import traceability, identity_pair_scores
from tgrepair.application_queries import audit_queries

ROOT = Path(__file__).resolve().parents[1]


def damage_counts(original, repaired, reference, types, clusters):
    """Count direct assignment and full alias-expansion damage separately."""
    before = {e.uid: e for e in original.events}
    identity = reference['identity']
    cluster_truth = defaultdict(set)
    for handle, cluster in clusters.items():
        if original.objects[handle] in types:
            cluster_truth[cluster].add(identity[handle])
    counts = {'originally_correct_events': 0, 'originally_incorrect_events': 0,
              'direct_new_damage': 0, 'direct_repaired_events': 0,
              'expanded_new_damage': 0, 'expanded_repaired_events': 0}
    for event in repaired.events:
        old = before[event.uid]
        if (old.time, old.kind, old.attributes) != (event.time, event.kind, event.attributes):
            raise AssertionError('Protected event content changed')
        truth = {o for o in reference['event_participation'][event.uid] if original.objects[o] in types}
        prior = {identity[o] for o in old.participants if original.objects[o] in types}
        direct = {identity[o] for o in event.participants if original.objects[o] in types}
        expanded = set().union(*(cluster_truth[clusters[o]] for o in event.participants
                                 if original.objects[o] in types))
        if prior == truth:
            counts['originally_correct_events'] += 1
            counts['direct_new_damage'] += int(direct != truth)
            counts['expanded_new_damage'] += int(expanded != truth)
        else:
            counts['originally_incorrect_events'] += 1
            counts['direct_repaired_events'] += int(direct == truth)
            counts['expanded_repaired_events'] += int(expanded == truth)
    return counts


def development_task(domain, seed, variant='mixed'):
    directory = ROOT / 'revision/data/development_tasks_v1' / domain / f'{variant}_seed{seed}'
    if directory.exists():
        task = json.loads((directory / 'task.json').read_text())
        for key, name in [('graph_sha256', 'observed.json.gz'), ('reference_sha256', 'reference.json')]:
            if sha256(directory / name) != task[key]:
                raise ValueError('Development task changed')
        return directory, task
    source = ROOT / 'revision/data/cohorts_v1' / domain / 'development/reference_graph.json.gz'
    manifest = json.loads((ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json').read_text())
    clean = load_graph(source)
    types = set(manifest['target_types'])
    if variant == 'mixed':
        observed, reference = mixed_faults(clean, 123000 + seed, types)
    elif variant == 'clean':
        observed, reference = mixed_faults(clean, 123000 + seed, types, 0., 0., 0.)
    elif variant == 'pipeline':
        from tgrepair.pipeline_faults import pipeline_faults
        observed, reference = pipeline_faults(clean, 123000 + seed, types)
    else:
        raise ValueError(variant)
    directory.mkdir(parents=True)
    save_graph(observed, directory / 'observed.json.gz')
    (directory / 'reference.json').write_text(json.dumps(reference) + '\n')
    task = {'domain': domain, 'partition': 'revision_development', 'variant': variant, 'seed': seed,
            'target_types': sorted(types), 'source_sha256': sha256(source),
            'graph_sha256': sha256(directory / 'observed.json.gz'),
            'reference_sha256': sha256(directory / 'reference.json')}
    (directory / 'task.json').write_text(json.dumps(task, indent=2) + '\n')
    return directory, task


def evaluate_development(domain, seed, variant='mixed'):
    training_path = ROOT / 'revision/reports' / f'budget_training_{domain}_seed{seed}_v1.json'
    training = json.loads(training_path.read_text())
    if training['status'] != 'completed_revision_budget_training':
        raise ValueError('Finish declared fitting before evaluation')
    directory, task = development_task(domain, seed, variant)
    graph = load_graph(directory / 'observed.json.gz')
    reference = json.loads((directory / 'reference.json').read_text())
    types = set(task['target_types'])
    out = ROOT / 'revision/data/decoding_v1' / f'development_{domain}_{variant}_seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'revision/reports' / f'development_budget_{domain}_{variant}_seed{seed}_v1.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    report = {'status': 'running_revision_development', 'task': task,
              'training_report_sha256': sha256(training_path),
              'evaluation_implementation_sha256': sha256(__file__),
              'before': traceability(graph, reference, types),
              'queries_before': audit_queries(graph, reference, types), 'results': []}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    start = time.monotonic()
    with threadpool_limits(limits=4):
        for row in training['models']:
            model_path = ROOT / 'revision/checkpoints' / (row['name'] + '.joblib')
            if sha256(model_path) != row['model_sha256']:
                raise ValueError('Checkpoint changed')
            scorer = FeatureScorer(model_path)
            state, audit = decode(graph, types, scorer, row['thresholds'], rounds=4,
                                  participant_budget=128, pair_budget=64,
                                  progress=lambda r: print(json.dumps({'model': row['name'], **r}), flush=True))
            result = {'name': row['name'], 'regime': row['regime'],
                      'total_verified_root_bundles': row['total_verified_root_bundles'],
                      'thresholds': row['thresholds'], 'model_sha256': row['model_sha256'],
                      'history': traceability(state.graph, reference, types, state.clusters),
                      'identity': identity_pair_scores(state.graph, reference, types, state.clusters),
                      'events': damage_counts(graph, state.graph, reference, types, state.clusters),
                      'queries': audit_queries(state.graph, reference, types, state.clusters),
                      'edits': audit['edit_count'], 'seconds': audit['seconds']}
            model_out = out / row['name']
            model_out.mkdir()
            save_graph(state.graph, model_out / 'repaired.json.gz')
            (model_out / 'clusters.json').write_text(json.dumps(state.clusters) + '\n')
            (model_out / 'audit.json').write_text(json.dumps(audit) + '\n')
            result['audit_sha256'] = sha256(model_out / 'audit.json')
            report['results'].append(result)
            report['seconds'] = time.monotonic() - start
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'model': row['name'], 'f1': result['history']['f1'],
                              'events': result['events'], 'seconds': result['seconds']}), flush=True)
    report['status'] = 'completed_revision_development'
    report_path.write_text(json.dumps(report, indent=2) + '\n')

