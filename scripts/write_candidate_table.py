"""Render the complete initial candidate audit independently of model outcomes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256
from write_confirmation_tables import table, NAMES


def main():
    path = ROOT/'reports/confirmation_candidate_evidence_v1.json'
    report = json.loads(path.read_text())
    assert report['status'] == 'complete_candidate_analysis'
    for name, digest in report['source_reports'].items():
        assert sha256(ROOT/name) == digest
    groups = {(x['domain'], x['variant'], x['metric']): x['summary'] for x in report['groups']}
    rows = []
    for domain in ['bpi2017', 'bpi2019', 'bpi2012_check']:
        metrics = sorted(x[2] for x in groups if x[0] == domain and x[1] == 'mixed' and x[2].startswith('missing_'))
        metrics += ['identity_all', 'identity_both_observed']
        for metric in metrics:
            label = {'identity_all': 'Identity pair: all', 'identity_both_observed': 'Identity pair: both observed'}.get(metric, metric.replace('missing_', 'Missing '))
            row = [NAMES[domain], label]
            for variant in ['mixed', 'pipeline']:
                value = groups[domain, variant, metric]
                row.append(f"{value['mean']:.3f} [{value['min']:.3f}, {value['max']:.3f}]")
            rows.append(row)
    content = table(['Source', 'Candidate target', 'Mixed', 'Pipeline'], rows,
        'Initial candidate recall, mean and observed range over three fault trials. Missing participation is audited on 2,048 uniformly sampled events per task; identity coverage enumerates candidate pairs. Both-observed identity recall excludes pairs with a handle lacking an observed history. Coverage measures retrievability, not identifiability or final recovery.',
        'tab:candidate-coverage')
    output = ROOT/'paper/tables/confirmation_candidates.tex'
    output.parent.mkdir(exist_ok=True)
    output.write_text(content)
    output.with_suffix('.sources.json').write_text(json.dumps({'analysis_hash': sha256(path), 'script_hash': sha256(__file__)}, indent=2)+'\n')


if __name__ == '__main__':
    main()
