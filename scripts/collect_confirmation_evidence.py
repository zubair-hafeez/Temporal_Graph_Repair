"""Read-only result validation and aggregation; never select or alter a model."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256
from run_confirmation_suite import jobs

SUITES = ('primary', 'clean', 'pipeline', 'source', 'adaptation', 'structural', 'controls')


def ratio(a, b):
    return a / b if b else None


def summarize(values):
    """An undefined precision stays undefined, rather than becoming zero."""
    present = [x for x in values if x is not None]
    return {'values': values, 'defined_trials': len(present),
            'mean': statistics.mean(present) if present else None,
            'min': min(present) if present else None,
            'max': max(present) if present else None}


def flat_result(row, report):
    history = report['after']['history']
    identity = report['after']['identity']
    events = report['after']['events']
    result = {k: row[k] for k in ('domain', 'view', 'variant', 'seed', 'arm', 'output')}
    result.update(suite=row['suite'], model=report['config']['model'],
                  before_f1=report['before']['f1'], after_f1=history['f1'],
                  gain_f1=history['f1']-report['before']['f1'],
                  before_exact=report['before']['exact_history_fraction'],
                  after_exact=history['exact_history_fraction'],
                  history_precision=history['precision'], history_recall=history['recall'],
                  target_entities=history['entities'],
                  identity_precision=identity['precision'], identity_recall=identity['recall'],
                  identity_predicted_pairs=identity['predicted_pairs'],
                  identity_true_pairs=identity['true_pairs'],
                  identity_correct_pairs=identity['correct_pairs'],
                  identity_false_pairs=identity['predicted_pairs']-identity['correct_pairs'],
                  originally_correct_events=events['originally_correct_events'],
                  preserved_correct_events=events['preserved_correct_events'],
                  newly_damaged_events=events['originally_correct_events']-events['preserved_correct_events'],
                  correct_event_preservation=ratio(events['preserved_correct_events'], events['originally_correct_events']),
                  damaged_events=events['damaged_events'], repaired_events=events['repaired_events'],
                  damaged_event_recovery=ratio(events['repaired_events'], events['damaged_events']),
                  edits=report['edit_count'], seconds=report['seconds'])
    for phase, key in [('before', 'queries_before'), ('after', 'queries_after')]:
        query = report[key]
        for field in ['empty_history_fraction', 'first_timestamp_correct_fraction', 'last_timestamp_correct_fraction']:
            result[phase+'_'+field] = query[field]
        for name in ['cancellation_evidence', 'exception_evidence']:
            result[phase+'_'+name+'_f1'] = query['queries'][name]['f1']
    return result


def collect(root, protocol_path, domains, suites):
    protocol = json.loads(protocol_path.read_text())
    digest = sha256(protocol_path)
    coverage, rows, per_type, pins = [], [], [], {}
    expected = []
    for suite in suites:
        expected.extend(dict(row, suite=suite) for row in jobs(protocol, domains, suite))
    for row in expected:
        path = root/'reports'/f"{row['output']}.json"
        status = {'output': row['output'], 'domain': row['domain'], 'suite': row['suite']}
        if not path.exists():
            coverage.append(dict(status, status='not_started')); continue
        report = json.loads(path.read_text())
        if report.get('pins', {}).get('protocol_hash') != digest:
            raise ValueError('Report protocol mismatch: '+str(path))
        if report['status'] != 'completed_confirmation_evaluation':
            coverage.append(dict(status, status=report['status'])); continue
        task_path = root/row['task']/'task.json'
        if report['pins']['task_hash'] != sha256(task_path):
            raise ValueError('Report task mismatch: '+str(path))
        if report['task']['domain'] != row['domain'] or report['config']['task'] != row['task']:
            raise ValueError('Report belongs to a different task: '+str(path))
        if row['model'] is not None and report['config']['model'] != row['model']:
            raise ValueError('Report model mismatch: '+str(path))
        if not report['after']['protected_event_fields_preserved']:
            raise ValueError('Protected event fields changed: '+str(path))
        flat = flat_result(row, report)
        rows.append(flat)
        for typ, after in report['after']['history']['per_type'].items():
            before = report['before']['per_type'][typ]
            per_type.append({k: flat[k] for k in ['domain', 'view', 'variant', 'seed', 'arm', 'suite', 'output']} |
                            {'type': typ, 'entities': after['entities'], 'before_f1': before['f1'],
                             'after_f1': after['f1'], 'gain_f1': after['f1']-before['f1'],
                             'before_exact': before['exact_history_fraction'],
                             'after_exact': after['exact_history_fraction']})
        pins[str(path.relative_to(root))] = sha256(path)
        coverage.append(dict(status, status='completed'))
    def group_key(row):
        arm = row['arm']
        if row['suite'] in {'primary', 'clean', 'pipeline'}:
            arm = arm.rsplit('_r', 1)[0]
        return row['domain'], row['suite'], arm
    expected_groups, completed_groups = defaultdict(list), defaultdict(list)
    for row in expected: expected_groups[group_key(row)].append(row)
    for row in rows: completed_groups[group_key(row)].append(row)
    groups = []
    numeric = [k for k in rows[0] if k not in {'domain', 'view', 'variant', 'seed', 'arm', 'output', 'suite', 'model'}] if rows else []
    for key, wanted in expected_groups.items():
        present = sorted(completed_groups[key], key=lambda r: (r['seed'], r['arm']))
        group = dict(zip(['domain', 'suite', 'arm'], key))
        group.update(expected_trials=len(wanted), completed_trials=len(present), outputs=[r['output'] for r in present])
        # Do not report a partial-trial average as the declared comparison.
        group['metrics'] = {field: summarize([r[field] for r in present]) for field in numeric} if len(present)==len(wanted) else None
        groups.append(group)
    paired = []
    for domain in domains:
        pairs = []
        for spec in protocol['primary_comparison']['pairs']:
            r = spec['replicate']; seed = spec['fault_seed']
            a = next((x for x in rows if x['domain']==domain and x['suite']=='primary' and x['arm']==f'plain_r{r}'), None)
            b = next((x for x in rows if x['domain']==domain and x['suite']=='primary' and x['arm']==f'varied_r{r}'), None)
            if a is None or b is None: continue
            if a['seed'] != seed or b['seed'] != seed or a['before_f1'] != b['before_f1']:
                raise ValueError('Paired comparison inputs differ')
            pairs.append({'replicate': r, 'fault_seed': seed, 'before_f1': a['before_f1'],
                          'original_f1': a['after_f1'], 'varied_f1': b['after_f1'],
                          'paired_gain': b['after_f1']-a['after_f1']})
        paired.append({'domain': domain, 'pairs': pairs,
                       'gain_summary': summarize([r['paired_gain'] for r in pairs]) if len(pairs)==3 else None})
    all_sources = {name: {'included_in_this_report': name in domains, 'role': spec['role'],
                          'source_file_available': Path(spec['path']).exists()}
                   for name, spec in protocol['sources'].items()}
    return {'status': 'complete_selected_confirmation' if all(r['status']=='completed' for r in coverage) else 'incomplete_confirmation',
            'protocol_hash': digest, 'domains': domains, 'suites': suites, 'coverage': coverage,
            'coverage_counts': dict(Counter(r['status'] for r in coverage)), 'all_predeclared_sources': all_sources,
            'trial_results': rows, 'per_type_results': per_type, 'groups': groups, 'primary_paired': paired,
            'source_reports': pins,
            'limits': ['Three primary trials jointly vary generated training worlds and test faults; their variance components are not separately estimated.',
                       'Events and entity histories are not treated as independent experimental replications.',
                       'Undefined query or identity precision is omitted from its descriptive mean; defined-trial counts and raw values are retained.',
                       'Runtime is wall time under shared concurrent load, not a controlled speed comparison.',
                       'Completion of selected domains does not imply completion of every predeclared source.']}


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='docs/CONFIRMATION_PROTOCOL_V1.json')
    parser.add_argument('--domains', nargs='+', required=True)
    parser.add_argument('--suites', nargs='+', choices=SUITES, default=list(SUITES))
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = collect(ROOT, ROOT/args.protocol, args.domains, args.suites)
    report['analysis_script_hash'] = sha256(__file__)
    if args.require_complete and report['status']!='complete_selected_confirmation':
        raise RuntimeError('Requested confirmation evidence is incomplete: '+str(report['coverage_counts']))
    destination = ROOT/'reports'/args.output
    destination.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    write_csv(destination.with_name(destination.name+'_trials.csv'), report['trial_results'])
    write_csv(destination.with_name(destination.name+'_types.csv'), report['per_type_results'])
    print(json.dumps({'status': report['status'], 'coverage': report['coverage_counts']}))


if __name__ == '__main__': main()
