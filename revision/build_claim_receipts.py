"""Record exact saved-source paths for numerical manuscript claims; no fitting."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tex = '\n'.join(p.read_text() for p in (ROOT / 'revision/paper').glob('*.tex'))
    source = 'revision/reports/coupled_summary_v1.json'
    summary = json.loads((ROOT / source).read_text())
    claims = []

    def add(name, source, path, literal, note='Exact source value; displayed rounding checked separately.'):
        value = json.loads((ROOT / source).read_text())
        for key in path:
            value = value[key]
        assert literal in tex, (name, literal)
        claims.append(dict(id=name, source=source, source_sha256=digest(ROOT / source),
                           field_path=path, value=value, manuscript_literal=literal, note=note))

    def group(domain, regime, budget, metric, literal, variant='mixed', point='primary95', stat='mean'):
        matches = [i for i, g in enumerate(summary['groups']) if
                   (g['domain'], g['regime'], g['budget'], g['variant'], g['operating_point']) ==
                   (domain, regime, budget, variant, point)]
        assert len(matches) == 1
        i = matches[0]
        add(f'{domain}_{regime}_{budget}_{variant}_{point}_{metric}_{stat}_{literal}', source,
            ['groups', i, 'metrics', metric, stat], literal)
        return summary['groups'][i]['metrics'][metric][stat]

    # Every primary-budget table value, including calibration-only failures.
    for g in summary['groups']:
        if g['variant'] == 'mixed' and g['operating_point'] == 'primary95' and g['budget'] > 0:
            literal = f"{100*g['metrics']['history.f1']['mean']:.1f}"
            group(g['domain'], g['regime'], g['budget'], 'history.f1', literal)

    for d, r, b, literal in [('bpi2019','mixed_varied',64,'.856'),
                            ('bpi2019','real_only',256,'.845')]:
        value = group(d,r,b,'history.f1',literal)
        assert f'{value:.3f}'.lstrip('0') == literal
    for d, r, b, metric, literal, style in [
        ('bpi2019','real_only',64,'history.exact_history_fraction','36.5','percent'),
        ('bpi2019','mixed_varied',64,'history.exact_history_fraction','48.7','percent'),
        ('bpi2019','real_only',64,'events.expanded_repaired_events','2,992','integer'),
        ('bpi2019','mixed_varied',64,'events.expanded_repaired_events','3,880','integer'),
        ('bpi2019','real_only',64,'events.expanded_new_damage','411','integer'),
        ('bpi2019','mixed_varied',64,'events.expanded_new_damage','237','integer'),
        ('bpi2019','real_only',64,'queries.queries.cancellation_evidence.f1','89.3','percent'),
        ('bpi2019','mixed_varied',64,'queries.queries.cancellation_evidence.f1','91.1','percent'),
        ('bpi2017','real_only',64,'queries.queries.cancellation_evidence.f1','79.4','percent'),
        ('bpi2017','mixed_original',64,'queries.queries.cancellation_evidence.f1','87.3','percent'),
        ('bpi2017','real_only',64,'queries.first_timestamp_correct_fraction','82.5','percent'),
        ('bpi2017','mixed_original',64,'queries.first_timestamp_correct_fraction','78.6','percent'),
        ('bpi2019','mixed_varied',256,'identity.precision','.912','fraction'),
        ('bpi2019','mixed_varied',256,'identity.recall','.378','fraction'),
    ]:
        value = group(d,r,b,metric,literal)
        rendered = f'{100*value:.1f}' if style == 'percent' else (f'{value:,.0f}' if style == 'integer' else f'{value:.3f}'.lstrip('0'))
        assert rendered == literal
    group('bpi2017','calibration_only_varied',16,'events.expanded_new_damage','9,028',stat='maximum')
    assert group('bpi2017','calibration_only_varied',16,'events.expanded_new_damage','9,028',stat='maximum') == 9028

    for domain in ['bpi2017','bpi2019']:
        for regime in ['generated_original','generated_varied']:
            value = next(g['metrics']['history.f1']['mean'] for g in summary['groups'] if
                         (g['domain'],g['variant'],g['regime']) == (domain,'mixed',regime))
            group(domain,regime,0,'history.f1',f'{value*100:.1f}')
        for regime in ['real_only','mixed_varied']:
            for point in ['primary95','strict99']:
                for metric, scale in [('history.f1',100),('events.expanded_new_damage',1),('events.expanded_repaired_events',1)]:
                    g = next(g for g in summary['groups'] if (g['domain'],g['variant'],g['regime'],g['budget'],g['operating_point']) == (domain,'mixed',regime,64,point))
                    group(domain,regime,64,metric,f"{scale*g['metrics'][metric]['mean']:.1f}",point=point)
        for regime in ['real_only','mixed_original','mixed_varied']:
            for budget in [16,64]:
                if domain == 'bpi2017':
                    values = group(domain,regime,budget,'identity.recall','recover no true alias pairs',stat='values')
                    assert values == [0.,0.,0.]

    comparison = 'revision/reports/final_comparison_checks_v1.json'
    c = json.loads((ROOT/comparison).read_text())['comparisons']
    for name,literal,wins in [('purchasing_varied64_vs_real256','in all three paired trials',3),
                              ('purchasing_varied64_vs_generated_varied0','in just one of three trials',1)]:
        values = c[name]['per_seed_difference']
        assert sum(v>0 for v in values) == wins
        add(name,comparison,['comparisons',name,'per_seed_difference'],literal,
            f'Paired positive differences: {wins}/3. Full list retained, not an inferential significance claim.')

    # Native coverage checked directly in all six saved native control reports.
    for domain in ['bpi2017','bpi2019']:
        values = []
        for seed in range(3):
            native = f'revision/reports/native_controls_confirmation_{domain}_seed{seed}_missing30_roots2048_v1.json'
            value = json.loads((ROOT/native).read_text())['candidate_recall']
            assert value > .85
            values.append(value)
            add(f'native_coverage_{domain}_{seed}',native,['candidate_recall'],r'Candidate coverage exceeds 85\%',
                'All six individual candidate-recall values exceed .85; mean and range also verified.')
        expected = ('98.1','98.1','98.2') if domain == 'bpi2017' else ('86.0','85.9','86.0')
        assert tuple(f'{100*x:.1f}' for x in [mean(values),min(values),max(values)]) == expected

    # Keep unique claim IDs if the same numerical claim was checked twice.
    claims = list({c['id']:c for c in claims}.values())
    output = {'status':'numerical_claim_receipts_verified','script_sha256':digest(Path(__file__)),
              'scope':'Saved-source numerical traceability and explicitly asserted rounding/paired comparisons; not external peer review.',
              'claims':claims}
    (ROOT/'revision/reports/numerical_claims_v1.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'status':output['status'],'claims':len(claims)}))


if __name__ == '__main__':
    main()
