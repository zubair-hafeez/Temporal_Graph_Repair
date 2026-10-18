"""Create the final coupled-repair declaration without evaluating outcomes.

This is a one-time prospective declaration, not a model-selection routine.
All fitted models and selected thresholds already exist from development.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.budget_learning import select_thresholds
from tgrepair.data import sha256


def freeze():
    destination = ROOT / 'revision/docs/COUPLED_CONFIRMATION_PROTOCOL_V1.json'
    if destination.exists():
        raise FileExistsError('Never replace a frozen protocol')
    for directory in ['revision/reports/coupled_confirmation_v1',
                      'revision/data/confirmation_tasks_v1', 'revision/data/coupled_decoding_v1']:
        if (ROOT / directory).exists() and any((ROOT / directory).iterdir()):
            raise RuntimeError('Coupled confirmation has already been accessed')
    dependencies = ['revision/confirmation.py', 'revision/evaluation.py',
                    'revision/annotation.py', 'revision/budget_learning.py',
                    'revision/extra_controls.py', 'revision/cohorts.py',
                    'revision/freeze_coupled.py']
    dependencies += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'tgrepair').glob('*.py'))]
    protocol = {
        'status': 'frozen_before_coupled_confirmation_outcomes',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'Do fixed generated supervision and target verification jointly improve offline operational history repair at matched incremental verification budgets?',
        'primary_comparisons': 'Both generated mixtures versus real-only at each of 16, 64 and 256 total verified histories, separately in both applications. Report every paired seed and mean/range; no pooled universal superiority claim or choice of best test model.',
        'development_decision': 'Both generator variants remain because development favors the original mixture for loans and the varied mixture for purchasing. Fixed mixture mass remains unchanged. Calibration-only and zero-label controls distinguish refitting from threshold adaptation. Atomic-root decoding stays a development-only alternative: its benefit is inconsistent relative to confidence tightening. Strict99 is a sensitivity operating point, not the primary method.',
        'known_evidence_before_freeze': 'All development budgets and controls are complete. Separate native missing-ID confirmation is complete and audited: generated scorers perform poorly on purchasing relative to target-local controls. Native data/outputs are disjoint from coupled confirmation; this declaration makes no claim of ignorance of native results.',
        'claim_limits': ['Previously studied source processes; fresh disjoint roots, not new institutions.',
                         'Injected recording faults and simulated complete verification, not independently verified natural business errors or measured annotation time.',
                         'Three seeds vary allocation/faults on fixed cohorts; these are not three independently sampled populations.',
                         'Historical model and workflow development labels are additional to incremental budgets.',
                         'Losses, harm, missing candidate coverage and all prespecified jobs must be retained. No model/threshold changes after this declaration.'],
        'domains': {}, 'fault_seed_base': 131000,
        'decoder': {'rounds': 4, 'participant_budget': 128, 'pair_budget': 64},
        'execution_shards': 6, 'task_shards': {}, 'jobs': [],
        'implementation': {p: sha256(ROOT / p) for p in dependencies},
        'inputs': {}, 'models': {},
        'analysis': {'primary': 'entity-and-alias-macro history F1',
                     'secondary': ['exact-history fraction', 'identity pair precision/recall', 'expanded newly damaged and repaired events', 'cancellation and temporal-boundary query recovery'],
                     'summary': 'Mean, observed range and paired seed differences. Do not count correlated events as independent replicates or infer population confidence from three runs.',
                     'robustness': 'Clean and independent pipeline faults for all three budget64 regimes and both zero-label generators; strict99 only on mixed faults at budget64 for real-only and mixed-varied.',
                     'completion': 'Every declared job must complete or an explicit failure be reported. No stopping after favorable outcomes.'}}

    def pin(path):
        path = str(path)
        protocol['inputs'][path] = sha256(ROOT / path)

    def add(domain, seed, variant, row, operating='primary95'):
        model = row.get('model_path', 'revision/checkpoints/' + row['name'] + '.joblib')
        digest = sha256(ROOT / model)
        if digest != row['model_sha256']:
            raise ValueError('Model differs from fitting evidence')
        protocol['models'][model] = digest
        job = {'id': f'{domain}_seed{seed}_{variant}_{row["regime"]}_{row["total_verified_root_bundles"]}_{operating}',
               'domain': domain, 'seed': seed, 'variant': variant,
               'regime': row['regime'], 'budget': row['total_verified_root_bundles'],
               'model_path': model, 'thresholds': row['thresholds'],
               'decoder': 'original', 'operating_point': operating}
        protocol['jobs'].append(job)

    for di, domain in enumerate(['bpi2017', 'bpi2019']):
        metadata = json.loads((ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json').read_text())
        protocol['domains'][domain] = {'root_type': metadata['root_type'], 'target_types': metadata['target_types']}
        for name in ['reference_graph.json.gz', 'bundles.json']:
            pin(f'revision/data/cohorts_v1/{domain}/confirmation/{name}')
        for seed in range(3):
            for variant in ['mixed', 'clean', 'pipeline']:
                protocol['task_shards'][f'{domain}/{seed}/{variant}'] = di * 3 + seed
            training_path = f'revision/reports/budget_training_{domain}_seed{seed}_v1.json'
            extra_path = f'revision/reports/extra_models_{domain}_seed{seed}_v1.json'
            pin(training_path); pin(extra_path)
            training = json.loads((ROOT / training_path).read_text())
            extra = json.loads((ROOT / extra_path).read_text())
            if training['status'] != 'completed_revision_budget_training' or extra['status'] != 'completed_extra_model_preparation':
                raise RuntimeError('Training or control preparation incomplete')
            pin(f'revision/data/annotation_v1/{domain}/seed{seed}/manifest.json')
            for row in training['models']:
                scores_path = f'revision/checkpoints/{row["name"]}_calibration.npz'
                pin(scores_path)
                if protocol['inputs'][scores_path] != row['calibration_sha256']:
                    raise ValueError('Calibration cache changed')
                add(domain, seed, 'mixed', row)
                if row['total_verified_root_bundles'] == 64:
                    for variant in ['clean', 'pipeline']:
                        add(domain, seed, variant, row)
                    if row['regime'] in ['real_only', 'mixed_varied']:
                        scores = np.load(ROOT / scores_path)
                        thresholds, _ = select_thresholds(scores['label'], scores['probability'], scores['operation'], precision_target=.99)
                        add(domain, seed, 'mixed', dict(row, thresholds=thresholds), 'strict99')
            for row in extra['models']:
                add(domain, seed, 'mixed', row)
                if row['total_verified_root_bundles'] == 0:
                    for variant in ['clean', 'pipeline']:
                        add(domain, seed, variant, row)
    for path in ['revision/docs/COHORT_PLAN_V1.json', 'revision/reports/development_summary_v1.json',
                 'revision/reports/native_confirmation_audit_v1.json']:
        pin(path)
    assert len(protocol['jobs']) == 156
    assert len({j['id'] for j in protocol['jobs']}) == 156
    destination.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'protocol': str(destination), 'sha256': sha256(destination), 'jobs': len(protocol['jobs'])}))


if __name__ == '__main__':
    freeze()
