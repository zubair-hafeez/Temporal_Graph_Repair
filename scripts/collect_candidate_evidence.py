"""Summarize all declared initial-retrieval audits, keeping fault trials separate."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def summary(values):
    return dict(values=values, mean=statistics.mean(values), min=min(values), max=max(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='confirmation_candidate_audits_real_v1')
    parser.add_argument('--output', default='confirmation_candidate_evidence_v1')
    args = parser.parse_args()
    suite_path = ROOT/'reports'/f'{args.suite}.json'
    suite = json.loads(suite_path.read_text())
    protocol_path = ROOT/'docs/CONFIRMATION_PROTOCOL_V1.json'
    protocol = json.loads(protocol_path.read_text())
    assert suite['status'] == 'completed_candidate_audit_suite'
    assert suite['protocol_hash'] == sha256(protocol_path)
    sources = {str(suite_path.relative_to(ROOT)): sha256(suite_path)}
    rows, grouped = [], {}
    for job in suite['jobs']:
        path = ROOT/'reports'/f"{job['output']}.json"
        result = json.loads(path.read_text())
        outcome = next(x for x in suite['outcomes'] if x['output'] == job['output'])
        assert outcome['status'] == 'completed' and outcome['hash'] == sha256(path)
        assert result['task_hash'] == sha256(ROOT/job['task']/'task.json')
        coverage = result['coverage']
        assert (coverage['sample_seed'], coverage['participant_budget'], coverage['pair_budget']) == (2027, 128, 64)
        values = {'identity_all': coverage['identity']['recall_all_true_pairs'],
                  'identity_both_observed': coverage['identity']['recall_both_observed'],
                  'cold_handles': coverage['target_catalog_handles_without_history']}
        for typ, counts in coverage['participation_by_type'].items():
            values['missing_'+typ] = counts['missing_participation_recall']
        for metric, value in values.items():
            row = {k: job[k] for k in ('domain', 'view', 'variant', 'seed', 'output')}
            row.update(metric=metric, value=value)
            rows.append(row)
            grouped.setdefault((job['domain'], job['view'], job['variant'], metric), []).append(row)
        sources[str(path.relative_to(ROOT))] = sha256(path)
    required = sorted(x['fault_seed'] for x in protocol['primary_comparison']['pairs'])
    groups = []
    for key, present in grouped.items():
        present.sort(key=lambda row: row['seed'])
        assert [x['seed'] for x in present] == required
        groups.append(dict(zip(('domain', 'view', 'variant', 'metric'), key)) |
                      {'summary': summary([x['value'] for x in present])})
    report = {'status': 'complete_candidate_analysis', 'protocol_hash': sha256(protocol_path),
              'script_hash': sha256(__file__), 'source_reports': sources, 'trial_results': rows,
              'groups': groups, 'limits': [
                  'Uniform sample of 2048 observed events per task; labels are used only after candidate construction.',
                  'Identity coverage enumerates initial candidate pairs. The both-observed condition excludes cold handles.',
                  'Coverage is retrievability, not identifiability. Later graph edits can change candidate coverage.',
                  'Mean and observed range are across three fault trials, not independent events.']}
    output = ROOT/'reports'/args.output
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'groups': len(groups), 'audits': len(suite['jobs'])}))


if __name__ == '__main__':
    main()
