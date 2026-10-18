"""Render verified-history costs directly from the six annotation manifests."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def build():
    inputs, rows = {}, []
    for domain, label in [('bpi2017', 'Loans'), ('bpi2019', 'Purchasing')]:
        manifests = []
        for seed in range(3):
            path = ROOT / 'revision/data/annotation_v1' / domain / f'seed{seed}/manifest.json'
            manifests.append(json.loads(path.read_text()))
            inputs[str(path.relative_to(ROOT))] = sha256(path)
        for budget in [16, 64, 256]:
            costs = []
            for field in ['verified_events', 'verified_target_entities', 'verified_alias_handles']:
                values = []
                for manifest in manifests:
                    record = manifest['annotation_budgets'][str(budget)]
                    if sum(record[role]['verified_root_bundles'] for role in ['fit', 'calibration']) != budget:
                        raise AssertionError('Budget does not include both fitting and calibration')
                    values.append(sum(record[role][field] for role in ['fit', 'calibration']))
                costs.append(f'{min(values):,}--{max(values):,}')
            rows.append(' & '.join([label, str(budget)] + costs) + r' \\')
    text = r'''\begin{table}[t]
\centering
\begin{tabular}{llrrr}
\toprule
Source & Bundles & Verified events & Target entities & Alias handles \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\caption{Total simulated verification access, including fitting and calibration. Ranges cover the three declared allocation/fault seeds. A complete bundle includes associated target children and all their aliases. Counts are not measured annotator time; query subsampling does not reduce the reported verification cost.}
\label{tab:annotation-cost}
\end{table}
'''
    output = ROOT / 'revision/paper/annotation_costs.tex'
    output.write_text(text)
    report = {'status': 'annotation_cost_table_generated', 'script_sha256': sha256(__file__),
              'inputs': inputs, 'output': str(output.relative_to(ROOT)), 'output_sha256': sha256(output)}
    (ROOT / 'revision/reports/annotation_table_sources_v1.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    build()
