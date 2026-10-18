"""Fixed 64-bundle, seed-zero retraining spot check for all three regimes.

This checks fitting from saved, identified features on the same runtime. It is
not a claim of independent-hardware replication or full raw-feature rebuilding.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from revision.budget_learning import fit_operation, generated_examples, select_thresholds
from tgrepair.data import sha256


def replay(domain, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source_report = ROOT / 'revision/reports' / f'budget_training_{domain}_seed0_v1.json'
    training = json.loads(source_report.read_text())
    if training['status'] != 'completed_revision_budget_training':
        raise ValueError('Original fitting must be complete')
    for path, digest in training['implementation'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Fitting implementation changed')
    cache = ROOT / 'revision/data/annotation_v1' / domain / 'seed0'
    manifest = json.loads((cache / 'manifest.json').read_text())
    if sha256(cache / 'manifest.json') != training['annotation_manifest_sha256']:
        raise ValueError('Annotation manifest differs')
    for role in ['fit', 'calibration']:
        if sha256(cache / f'{role}_64.npz') != manifest['annotation_budgets']['64'][role]['sha256']:
            raise ValueError('Annotation features differ')
    real = np.load(cache / 'fit_64.npz')
    valid = np.load(cache / 'calibration_64.npz')
    result = {'status': 'running_retraining_spot_check', 'domain': domain, 'seed': 0, 'budget': 64,
              'script_sha256': sha256(__file__), 'original_report_sha256': sha256(source_report),
              'generated_cache_inventory_sha256': sha256(ROOT / 'revision/reports/generated_feature_cache_inventory_v1.json'),
              'scope': 'Same-runtime refitting from saved features, all three regimes. Does not certify independent-hardware retraining or raw-to-feature regeneration.',
              'models': []}
    names = {'mixed_original': 'edit_train_v2', 'mixed_varied': 'edit_train_attributes_v3'}
    start = time.monotonic()
    with threadpool_limits(limits=4):
        for original in training['models']:
            if original['total_verified_root_bundles'] != 64:
                continue
            regime = original['regime']
            generated = generated_examples(names[regime]) if regime in names else None
            models = []
            prediction = np.zeros(len(valid['y']))
            for op in range(3):
                mask = real['operation'] == op
                sx = sy = None
                if generated is not None:
                    gx, gy, gop = generated
                    sx, sy = gx[gop == op], gy[gop == op]
                model, _ = fit_operation(real['x'][mask], real['y'][mask], sx, sy, 0)
                models.append(model)
                selected = valid['operation'] == op
                if selected.any():
                    prediction[selected] = model.predict_proba(valid['x'][selected])[:, 1]
            model_path = output / (original['name'] + '.joblib')
            joblib.dump(models, model_path)
            old_scores = np.load(ROOT / 'revision/checkpoints' / (original['name'] + '_calibration.npz'))
            if not np.array_equal(valid['y'], old_scores['label']) or not np.array_equal(valid['operation'], old_scores['operation']):
                raise AssertionError('Calibration label/order mismatch')
            thresholds, _ = select_thresholds(valid['y'], prediction, valid['operation'])
            old_decisions = old_scores['probability'] >= np.asarray(original['thresholds'])[valid['operation']]
            new_decisions = prediction >= np.asarray(thresholds)[valid['operation']]
            row = {'name': original['name'], 'checkpoint_sha256': sha256(model_path),
                   'checkpoint_bytes_identical': sha256(model_path) == original['model_sha256'],
                   'calibration_predictions_identical': bool(np.array_equal(prediction, old_scores['probability'])),
                   'maximum_prediction_difference': float(np.max(np.abs(prediction - old_scores['probability']))),
                   'thresholds_identical': thresholds == original['thresholds'],
                   'threshold_maximum_difference': float(np.max(np.abs(np.asarray(thresholds) - original['thresholds']))),
                   'calibration_decisions_changed': int(np.count_nonzero(new_decisions != old_decisions)),
                   'calibration_predictions_close_at_1e_12': bool(np.allclose(prediction, old_scores['probability'], rtol=0, atol=1e-12))}
            result['models'].append(row)
            (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            if not row['calibration_predictions_close_at_1e_12'] or row['calibration_decisions_changed'] or row['threshold_maximum_difference'] > 1e-12:
                result['status'] = 'retraining_difference_requires_investigation'
                (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
                raise AssertionError('Retraining spot check differs; investigate without editing original evidence')
    result['status'] = 'selected_models_numerically_reproduced_with_identical_calibration_decisions'
    result['limitation'] = 'Exact byte/score equality failed in the initial audit at approximately 1e-15. This follow-up reports exact equality separately and checks absolute score/threshold tolerance 1e-12 and exact calibration decisions. Full iterative decoding replay is not established by this fitting check.'
    result['seconds'] = time.monotonic() - start
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    replay(args.domain, args.output_directory)
