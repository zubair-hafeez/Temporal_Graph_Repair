"""Build tables/figures from the entire audited, frozen coupled suite."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.confirmation import verify_protocol
from tgrepair.data import sha256

LABELS = {'real_only': 'Real only', 'mixed_original': 'Real + original generated',
          'mixed_varied': 'Real + varied generated', 'calibration_only_varied': 'Calibration only',
          'generated_original': 'Generated original', 'generated_varied': 'Generated varied'}
DOMAINS = {'bpi2017': 'Loans', 'bpi2019': 'Purchasing'}


def nested(row, key):
    for part in key.split('.'):
        row = row[part]
    return row


def stats(rows, key):
    values = [nested(r, key) for r in rows]
    if any(v is None for v in values):
        return None
    return {'mean': statistics.mean(values), 'minimum': min(values), 'maximum': max(values), 'values': values}


def format_metric(rows, key='history.f1', percent=True, ranges=False):
    values = stats(rows, key)
    if values is None:
        return '--'
    scale = 100 if percent else 1
    result = f'{values["mean"] * scale:.1f}'
    if ranges:
        result += f' ({values["minimum"] * scale:.1f}--{values["maximum"] * scale:.1f})'
    return result


def table(caption, label, columns, header, rows):
    return '\n'.join([r'\begin{table}[t]', r'\centering\small',
                      '\\caption{' + caption + '}', '\\label{' + label + '}',
                      '\\begin{tabular}{' + columns + '}', r'\toprule',
                      header + r' \\', r'\midrule',
                      *[r + r' \\' for r in rows], r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def build():
    protocol = verify_protocol()
    audit_path = ROOT / 'revision/reports/coupled_confirmation_audit_v1.json'
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'all_coupled_confirmation_predictions_and_edit_replays_verified':
        raise RuntimeError('Audit the complete suite first')
    groups, all_rows = defaultdict(list), []
    for job in protocol['jobs']:
        path = ROOT / 'revision/reports/coupled_confirmation_v1' / (job['id'] + '.json')
        if sha256(path) != audit['inputs'][path.relative_to(ROOT).as_posix()]:
            raise ValueError('Audited result changed')
        row = json.loads(path.read_text())
        groups[(job['domain'], job['variant'], job['regime'], job['budget'], job['operating_point'])].append(row)
        all_rows.append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r['pins']['job']['seed'])
        if [r['pins']['job']['seed'] for r in rows] != [0, 1, 2]:
            raise AssertionError('Every declared condition requires all three seeds')

    def get(domain, regime, budget, variant='mixed', operating='primary95'):
        return groups[(domain, variant, regime, budget, operating)]

    paper = ROOT / 'revision/paper'
    rows = []
    for domain in DOMAINS:
        for budget in [16, 64, 256]:
            rows.append(' & '.join([DOMAINS[domain], str(budget)] + [format_metric(get(domain, regime, budget)) for regime in ['real_only', 'mixed_original', 'mixed_varied', 'calibration_only_varied']]))
    text = table('Fresh combined-repair confirmation. Mean history F1 (percent) over three allocation/fault seeds. Budgets count fitting and calibration together; Fig.~\\ref{fig:budget-curves} shows observed ranges.',
                 'tab:coupled-budget', 'llrrrr', 'Application & Budget & Real only & + Original & + Varied & Calibration', rows)
    text += '\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/budget_curves.pdf}\n\\caption{History recovery against complete verified-history budget. Lines show three-seed means; bands show observed minima and maxima, not confidence intervals. Dotted lines show fixed generated scorers without new target verification.}\n\\label{fig:budget-curves}\n\\end{figure}\n'
    (paper / 'budget_results.tex').write_text(text)

    rows = []
    for domain in DOMAINS:
        for regime, budget in [('real_only', 64), ('mixed_original', 64), ('mixed_varied', 64), ('generated_original', 0), ('generated_varied', 0)]:
            clean = get(domain, regime, budget, 'clean')
            pipeline = get(domain, regime, budget, 'pipeline')
            rows.append(' & '.join([DOMAINS[domain], LABELS[regime], format_metric(clean),
                        format_metric(clean, 'events.expanded_new_damage', percent=False), format_metric(pipeline),
                        format_metric(pipeline, 'events.expanded_new_damage', percent=False)]))
    (paper / 'robustness_results.tex').write_text(table(
        'Mean history F1 (percent) and newly damaged event counts under clean inputs and independent pipeline faults. Adapted models use 64 verified bundles; generated-only models use zero new target bundles.',
        'tab:robustness', 'llrrrr', 'Application & Method & Clean F1 & Damage & Pipeline F1 & Damage', rows))

    rows = []
    for domain in DOMAINS:
        for regime in ['real_only', 'mixed_varied']:
            primary = get(domain, regime, 64)
            strict = get(domain, regime, 64, operating='strict99')
            rows.append(' & '.join([DOMAINS[domain], LABELS[regime], format_metric(primary), format_metric(strict),
                format_metric(primary, 'events.expanded_new_damage', False), format_metric(strict, 'events.expanded_new_damage', False),
                format_metric(primary, 'events.expanded_repaired_events', False), format_metric(strict, 'events.expanded_repaired_events', False)]))
    (paper / 'reliability_results.tex').write_text(table(
        'Threshold sensitivity at budget 64. Columns compare .95/.99 empirical calibration precision targets; neither is a guarantee on new histories. F1 is percent; damage and repaired-event counts are means over three seeds.',
        'tab:reliability', 'llrrrrrr', 'Application & Method & F1 .95 & .99 & Damage .95 & .99 & Repaired .95 & .99', rows))

    measures = ['history.f1', 'history.exact_history_fraction', 'before.f1', 'events.expanded_new_damage',
                'events.expanded_repaired_events', 'identity.precision', 'identity.recall',
                'queries.first_timestamp_correct_fraction', 'queries.last_timestamp_correct_fraction',
                'queries.queries.cancellation_evidence.f1']
    summary = {'status': 'complete_audited_coupled_summary', 'audit_sha256': sha256(audit_path),
               'script_sha256': sha256(__file__), 'groups': [], 'paired_differences': [],
               'source_reports': audit['inputs']}
    for key, values in groups.items():
        summary['groups'].append({'domain': key[0], 'variant': key[1], 'regime': key[2], 'budget': key[3],
            'operating_point': key[4], 'metrics': {m: stats(values, m) for m in measures}})
    for domain in DOMAINS:
        for budget in [16, 64, 256]:
            real = get(domain, 'real_only', budget)
            for regime in ['mixed_original', 'mixed_varied', 'calibration_only_varied']:
                alternative = get(domain, regime, budget)
                differences = [a['history']['f1'] - r['history']['f1'] for a, r in zip(alternative, real)]
                summary['paired_differences'].append({'domain': domain, 'budget': budget, 'regime': regime,
                    'seed_differences': differences, 'mean_difference': statistics.mean(differences),
                    'wins': sum(v > 0 for v in differences), 'losses': sum(v < 0 for v in differences)})
    (ROOT / 'revision/reports/coupled_summary_v1.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (ROOT / 'revision/reports/coupled_results_v1.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'domain', 'seed', 'variant', 'regime', 'budget', 'operating_point'] + measures)
        for row in all_rows:
            job = row['pins']['job']
            writer.writerow([job[k] for k in ['id', 'domain', 'seed', 'variant', 'regime', 'budget', 'operating_point']] + [nested(row, m) for m in measures])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    colors = ['#444444', '#0072B2', '#D55E00', '#009E73']
    for ax, domain in zip(axes, DOMAINS):
        for regime, color in zip(['real_only', 'mixed_original', 'mixed_varied', 'calibration_only_varied'], colors):
            values = [stats(get(domain, regime, budget), 'history.f1') for budget in [16, 64, 256]]
            ax.plot([16, 64, 256], [100 * v['mean'] for v in values], marker='o', color=color, label=LABELS[regime])
            ax.fill_between([16, 64, 256], [100 * v['minimum'] for v in values], [100 * v['maximum'] for v in values], color=color, alpha=.10)
        for regime, color in [('generated_original', colors[1]), ('generated_varied', colors[2])]:
            ax.axhline(100 * stats(get(domain, regime, 0), 'history.f1')['mean'], color=color, linestyle=':', linewidth=1)
        ax.set_xscale('log', base=4); ax.set_xticks([16, 64, 256], ['16', '64', '256'])
        ax.set_title(DOMAINS[domain]); ax.set_xlabel('Verified complete histories'); ax.grid(alpha=.18)
    axes[0].set_ylabel('History F1 (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .18, 1, 1))
    (paper / 'figures').mkdir(exist_ok=True)
    fig.savefig(paper / 'figures/budget_curves.pdf', bbox_inches='tight')
    fig.savefig(paper / 'figures/budget_curves.png', dpi=180, bbox_inches='tight')
    print(json.dumps({'status': summary['status'], 'jobs': len(all_rows)}))


if __name__ == '__main__':
    build()
