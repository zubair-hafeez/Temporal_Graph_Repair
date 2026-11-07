"""Derive manuscript tables and an exportable harm/recovery plot from audited data."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .run import HERE, write
from tgrepair.data import sha256


def main():
    report = json.loads((HERE / 'analysis.json').read_text())
    assert report['status'] == 'complete_all_156_followup_conditions_audited'
    rows = report['aggregates']
    def get(domain, budget=64, arm='real8_generated', mode='full', precision=.95, variant='mixed', safe=True):
        matches = [r for r in rows if r['domain'] == domain and r['budget'] == budget and r['arm'] == arm
                   and r['mode'] == mode and r['precision'] == precision and r['variant'] == variant and r['safe'] == safe]
        assert len(matches) == 1
        return matches[0]
    table = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Repeated-corruption follow-up: three-seed means on 128 reused roots per source. $R_8$ uses eight corruptions of each verified fitting set; $+G$ adds generated histories. $q$ is the calibration precision target. Recovery and damage count fully corrected initially incorrect events and newly incorrect initially correct events, with alias expansion. Equal $q$ does not guarantee equal test damage.}',
             r'\label{tab:repeated-corruption}', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{lrr rr rr rr}', r'\toprule',
             r' & & & \multicolumn{2}{c}{History F1 (\%)} & \multicolumn{2}{c}{Recovered} & \multicolumn{2}{c}{Damaged} \\',
             r'Source & Budget & $q$ & $R_8$ & $+G$ & $R_8$ & $+G$ & $R_8$ & $+G$ \\ \midrule']
    for domain, label in [('bpi2017', 'Loans'), ('bpi2019', 'Purchasing')]:
        for budget in [16, 64, 256]:
            for p in [.95, .99]:
                a, b = get(domain, budget, 'real8', precision=p), get(domain, budget, precision=p)
                table.append(f"{label if budget == 16 and p == .95 else ''} & {budget} & {p:.2f} & "
                             f"{a['f1']['mean']*100:.1f} & {b['f1']['mean']*100:.1f} & "
                             f"{a['recovery']['mean']:.0f} & {b['recovery']['mean']:.0f} & "
                             f"{a['damage']['mean']:.1f} & {b['damage']['mean']:.1f} \\")
        if domain == 'bpi2017': table.append(r'\midrule')
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    # Correct row endings independent of Python escaping in formatted literals.
    table = [line + '\\' if line.endswith('\\') and not line.endswith('\\\\') else line for line in table]
    (HERE / 'paper_table_primary.tex').write_text('\n'.join(table) + '\n')
    names = [('full', 'All information'), ('no_time', 'No timestamps'), ('no_context', 'No context objects'),
             ('no_attributes', 'No descriptions'), ('no_history', 'No history edges')]
    table = [r'\begin{table}[t]', r'\centering\small',
             r'\caption{Input ablations of $+G$, budget 64, $q=.95$: three-seed means. Information is removed before retrieval and fitting. No-context retains target root/child incidence; no-history removes incidence from the evidence index. These are information ablations, not architecture comparisons.}',
             r'\label{tab:followup-ablation}', r'\begin{tabular}{l rrr rrr}', r'\toprule',
             r' & \multicolumn{3}{c}{Loans} & \multicolumn{3}{c}{Purchasing} \\',
             r'Input & F1 (\%) & Recovered & Damaged & F1 (\%) & Recovered & Damaged \\ \midrule']
    for mode, name in names:
        a, b = get('bpi2017', mode=mode), get('bpi2019', mode=mode)
        table.append(f"{name} & {a['f1']['mean']*100:.1f} & {a['recovery']['mean']:.0f} & {a['damage']['mean']:.1f} & "
                     f"{b['f1']['mean']*100:.1f} & {b['recovery']['mean']:.0f} & {b['damage']['mean']:.1f} \\")
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    table = [line + '\\' if line.endswith('\\') and not line.endswith('\\\\') else line for line in table]
    (HERE / 'paper_table_ablation.tex').write_text('\n'.join(table) + '\n')
    (HERE / 'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True, constrained_layout=True)
    colors = {16: '#8451a1', 64: '#167f8e', 256: '#bd651a'}
    for ax, (domain, title) in zip(axes, [('bpi2017', 'Loans'), ('bpi2019', 'Purchasing')]):
        for budget in [16, 64, 256]:
            for arm, marker in [('real8', 'o'), ('real8_generated', '^')]:
                selected = [get(domain, budget, arm, precision=p) for p in [.99, .95]]
                x = [r['damage_rate']['mean'] * 1000 for r in selected]
                y = [r['recovery_rate']['mean'] * 100 for r in selected]
                ax.plot(x, y, color=colors[budget], alpha=.55, linewidth=1)
                ax.scatter(x, y, marker=marker, color=colors[budget], s=45,
                           facecolors='none' if arm == 'real8' else colors[budget],
                           label=f"{budget} bundles, {'R8' if arm == 'real8' else 'R8 + G'}")
        ax.set_title(title); ax.grid(alpha=.18)
        ax.set_xlabel('New damage per 1,000 initially correct events')
        ax.set_xlim(left=-.15)
    ymax = max(r['recovery_rate']['mean'] * 100 for r in rows
               if r['mode'] == 'full' and r['variant'] == 'mixed' and r['safe'])
    axes[0].set_ylim(0, ymax * 1.12)
    axes[0].set_ylabel('Initially incorrect events fully repaired (%)')
    axes[1].legend(fontsize=8.5, loc='lower right', frameon=False)
    fig.savefig(HERE / 'figures/recovery_damage.pdf')
    fig.savefig(HERE / 'figures/recovery_damage.png', dpi=160)
    plt.close(fig)
    write(HERE / 'presentation.json', {'analysis_sha256': sha256(HERE / 'analysis.json'),
                                      'script_sha256': sha256(__file__),
                                      'outputs': {name: sha256(HERE / name) for name in ['paper_table_primary.tex', 'paper_table_ablation.tex',
                                                  'figures/recovery_damage.pdf', 'figures/recovery_damage.png']}})


if __name__ == '__main__':
    main()
