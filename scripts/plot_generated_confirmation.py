"""Draw complete generated confirmation, with trial variation and preservation."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def main():
    path = ROOT/'reports/generated_confirmation_evidence_v1.json'
    evidence = json.loads(path.read_text())
    assert evidence['status'] == 'complete_generated_confirmation_analysis'
    for name, digest in evidence['source_reports'].items():
        assert sha256(ROOT/name) == digest
    summaries = evidence['summaries']
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    colors = ['#aab3b8', '#bd8a57', '#2d7f87']
    labels = ['Damaged', 'Original fields', 'Varied fields']
    for ax, metric, title in zip(axes, ['after_f1', 'after_exact'],
                                 ['History F1', 'Exactly restored histories']):
        base = summaries['plain']['mixed'][metric.replace('after_', 'before_')]
        data = [base, summaries['plain']['mixed'][metric], summaries['varied']['mixed'][metric]]
        for i, item in enumerate(data):
            ax.bar(i, item['mean'], color=colors[i], width=.62)
            ax.errorbar(i, item['mean'], yerr=[[item['mean']-item['min']], [item['max']-item['mean']]],
                        color='#23313c', capsize=5, lw=1)
            ax.scatter(i+np.linspace(-.12,.12,3), item['values'], s=16, color='#23313c', zorder=4)
            ax.text(i, item['max']+.026, f"{item['mean']:.3f}", ha='center', fontsize=10)
        ax.set_xticks(range(3), labels)
        ax.set_ylim(0, 1)
        ax.set_title(title, loc='left', fontsize=12, weight='bold')
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('96 held-out workflow worlds · three independent training collections', fontsize=12)
    destination = ROOT/'paper/figures/generated_confirmation'
    fig.savefig(destination.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(destination.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    sources = {'analysis': str(path.relative_to(ROOT)), 'analysis_hash': sha256(path),
               'script_hash': sha256(__file__),
               'display': 'Dots are training trials; whiskers are observed min/max, not confidence intervals.'}
    destination.with_name(destination.name+'_sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    print(destination)


if __name__ == '__main__':
    main()
