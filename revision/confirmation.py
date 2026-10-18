"""Reserved coupled-repair evaluation, disabled until its own protocol freeze.

Training, model selection and protocol creation are deliberately absent here.
Each completed result and each resumable original-decoder pass is tied to the
same task, model, thresholds, implementation and frozen job specification.
"""
import json
from pathlib import Path
import time

from threadpoolctl import threadpool_limits

from revision.evaluation import damage_counts
from tgrepair.application_queries import audit_queries
from tgrepair.corruption import mixed_faults
from tgrepair.data import load_graph, save_graph, sha256
from tgrepair.decoding import FeatureScorer, decode
from tgrepair.metrics import traceability, identity_pair_scores
from tgrepair.pipeline_faults import pipeline_faults

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'revision/docs/COUPLED_CONFIRMATION_PROTOCOL_V1.json'


def write_json(path, value):
    """Readers see a complete old or new report, never a truncated write."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_protocol():
    path = ROOT / PROTOCOL
    if not path.exists():
        raise RuntimeError('Coupled confirmation remains sealed: no frozen protocol')
    protocol = json.loads(path.read_text())
    if protocol['status'] != 'frozen_before_coupled_confirmation_outcomes':
        raise ValueError('Coupled confirmation protocol is not frozen')
    for group in ['implementation', 'inputs', 'models']:
        for name, expected in protocol[group].items():
            if sha256(ROOT / name) != expected:
                raise ValueError('Frozen coupled dependency changed: ' + name)
    ids = [job['id'] for job in protocol['jobs']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate coupled job identifier')
    return protocol


def prepare_task(domain, seed, variant, protocol):
    if not any(job['domain'] == domain and job['seed'] == seed and job['variant'] == variant
               for job in protocol['jobs']):
        raise ValueError('Undeclared confirmation task')
    source = ROOT / 'revision/data/cohorts_v1' / domain / 'confirmation/reference_graph.json.gz'
    spec = protocol['domains'][domain]
    expected = {'domain': domain, 'seed': seed, 'variant': variant,
                'fault_seed': protocol['fault_seed_base'] + seed,
                'source_sha256': sha256(source), 'target_types': spec['target_types'],
                'protocol_sha256': sha256(ROOT / PROTOCOL)}
    directory = ROOT / 'revision/data/confirmation_tasks_v1' / domain / f'{variant}_seed{seed}'
    if directory.exists():
        task = json.loads((directory / 'task.json').read_text())
        if task['pins'] != expected:
            raise ValueError('Confirmation task pins changed')
        if any(sha256(directory / name) != digest for name, digest in task['files'].items()):
            raise ValueError('Confirmation task content changed')
        return directory, task
    clean = load_graph(source)
    types = set(spec['target_types'])
    if variant == 'mixed':
        observed, reference = mixed_faults(clean, expected['fault_seed'], types)
    elif variant == 'clean':
        observed, reference = mixed_faults(clean, expected['fault_seed'], types, 0., 0., 0.)
    elif variant == 'pipeline':
        observed, reference = pipeline_faults(clean, expected['fault_seed'], types)
    else:
        raise ValueError(variant)
    directory.mkdir(parents=True)
    save_graph(observed, directory / 'observed.json.gz')
    write_json(directory / 'reference.json', reference)
    task = {'status': 'reserved_task_prepared', 'pins': expected,
            'files': {name: sha256(directory / name) for name in ['observed.json.gz', 'reference.json']}}
    write_json(directory / 'task.json', task)
    return directory, task


def evaluate_job(job, protocol):
    if job not in protocol['jobs']:
        raise ValueError('Undeclared coupled job or modified operating point')
    directory, task = prepare_task(job['domain'], job['seed'], job['variant'], protocol)
    destination = ROOT / 'revision/reports/coupled_confirmation_v1' / (job['id'] + '.json')
    output = ROOT / 'revision/data/coupled_decoding_v1' / job['id']
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / job['model_path']
    pins = {'job': job, 'protocol_sha256': sha256(ROOT / PROTOCOL),
            'task_sha256': sha256(directory / 'task.json'), 'model_sha256': sha256(model_path)}
    if model_path.relative_to(ROOT).as_posix() not in protocol['models']:
        raise ValueError('Unpinned model requested')
    if destination.exists():
        report = json.loads(destination.read_text())
        if report['pins'] != pins:
            raise ValueError('Existing job pins differ')
        if report['status'] == 'completed_coupled_confirmation':
            for name, digest in report['artifacts'].items():
                if sha256(output / name) != digest:
                    raise ValueError('Completed prediction artifact changed')
            print(json.dumps({'verified_completed_job': job['id']}), flush=True)
            return
    graph = load_graph(directory / 'observed.json.gz')
    reference = json.loads((directory / 'reference.json').read_text())
    types = set(task['pins']['target_types'])
    report = {'status': 'running_coupled_confirmation', 'pins': pins,
              'before': traceability(graph, reference, types),
              'queries_before': audit_queries(graph, reference, types)}
    write_json(destination, report)
    progress_path = output / 'progress.json'
    resume = None
    if progress_path.exists():
        saved = json.loads(progress_path.read_text())
        if saved['pins'] != pins or job['decoder'] != 'original':
            raise ValueError('Incompatible decoder progress')
        resume = saved['history']
    start = time.monotonic()
    scorer = FeatureScorer(model_path)
    kwargs = {'rounds': protocol['decoder']['rounds'],
              'participant_budget': protocol['decoder']['participant_budget'],
              'pair_budget': protocol['decoder']['pair_budget'],
              'progress': lambda r: print(json.dumps({'job': job['id'], **r}), flush=True)}
    with threadpool_limits(limits=4):
        if job['decoder'] == 'original':
            state, audit = decode(graph, types, scorer, job['thresholds'], **kwargs,
                resume_history=resume,
                checkpoint=lambda history: write_json(progress_path, {'pins': pins, 'history': history}))
        elif job['decoder'] == 'atomic_root':
            from revision.transactional_decoding import decode_atomic_root
            state, audit = decode_atomic_root(graph, types, scorer, job['thresholds'],
                                               protocol['domains'][job['domain']]['root_type'], **kwargs)
        else:
            raise ValueError('Unrecognized decoder')
    report.update({'history': traceability(state.graph, reference, types, state.clusters),
                   'identity': identity_pair_scores(state.graph, reference, types, state.clusters),
                   'events': damage_counts(graph, state.graph, reference, types, state.clusters),
                   'queries': audit_queries(state.graph, reference, types, state.clusters),
                   'edits': audit['edit_count'], 'this_execution_seconds': time.monotonic() - start,
                   'resumed_prior_passes': 0 if resume is None else len(resume)})
    save_graph(state.graph, output / 'repaired.json.gz')
    write_json(output / 'clusters.json', state.clusters)
    write_json(output / 'audit.json', audit)
    report['artifacts'] = {name: sha256(output / name) for name in ['repaired.json.gz', 'clusters.json', 'audit.json']}
    report['status'] = 'completed_coupled_confirmation'
    write_json(destination, report)
    print(json.dumps({'completed_job': job['id'], 'f1': report['history']['f1'],
                      'new_damage': report['events']['expanded_new_damage']}), flush=True)


def run_shard(shard, shards):
    protocol = verify_protocol()
    if shards != protocol['execution_shards'] or not 0 <= shard < shards:
        raise ValueError('Use the declared shard assignment to avoid overlapping task writers')
    # Every domain/seed/variant belongs wholly to one worker, so shared task
    # creation has no concurrent writers. Jobs remain in their frozen order.
    tasks = {'/'.join(map(str, (j['domain'], j['seed'], j['variant']))) for j in protocol['jobs']}
    assignment = protocol['task_shards']
    if set(assignment) != tasks or any(not isinstance(v, int) or not 0 <= v < shards for v in assignment.values()):
        raise ValueError('Incomplete or invalid frozen task assignment')
    for job in protocol['jobs']:
        key = '/'.join(map(str, (job['domain'], job['seed'], job['variant'])))
        if assignment[key] == shard:
            verify_protocol()
            evaluate_job(job, protocol)
    print(json.dumps({'completed_coupled_shard': shard, 'shards': shards}), flush=True)
