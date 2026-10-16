"""Matched total-annotation-budget learning with fixed generated supervision."""
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from threadpoolctl import threadpool_limits

from tgrepair.data import sha256

ROOT = Path(__file__).resolve().parents[1]


class AbstainingClassifier:
    """Absent-class fitting data provide no support for a repair operation."""
    def predict_proba(self, x):
        return np.column_stack([np.ones(len(x)), np.zeros(len(x))])


def generated_examples(name):
    import torch
    folder = ROOT / 'data/edit_learning' / name
    manifest = json.loads((folder / 'manifest.json').read_text())
    xs, ys, ops = [], [], []
    for record in manifest['records']:
        row = torch.load(folder / record['file'], map_location='cpu', weights_only=True)
        xs.append(row['observation']['action_features'].numpy())
        ys.append(row['labels'].numpy())
        ops.append(row['observation']['operations'].numpy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ops)


def select_thresholds(y, probability, operation, precision_target=.95, minimum_correct=5):
    thresholds, details = [], []
    for op in range(3):
        mask = operation == op
        labels, scores = y[mask], probability[mask]
        if not len(labels) or labels.sum() < minimum_correct:
            thresholds.append(1.1)
            details.append({'precision': None, 'recall': 0., 'reason': 'insufficient_verified_positive_actions'})
            continue
        precision, recall, values = precision_recall_curve(labels, scores)
        feasible = np.flatnonzero((precision[:-1] >= precision_target)
                                 & (recall[:-1] * labels.sum() >= minimum_correct))
        if not len(feasible):
            thresholds.append(1.1)
            details.append({'precision': None, 'recall': 0., 'reason': 'no_feasible_verified_operating_point'})
        else:
            i = feasible[np.argmax(recall[feasible])]
            thresholds.append(float(values[i]))
            details.append({'precision': float(precision[i]), 'recall': float(recall[i]),
                            'correct_actions': int(round(recall[i] * labels.sum()))})
    return thresholds, details


def fit_operation(real_x, real_y, synthetic_x=None, synthetic_y=None, seed=0):
    x, y = real_x, real_y
    weights = np.full(len(y), 1 / max(1, len(y)))
    if synthetic_x is not None:
        x = np.concatenate([x, synthetic_x])
        y = np.concatenate([y, synthetic_y])
        weights = np.concatenate([weights * .5, np.full(len(synthetic_y), .5 / max(1, len(synthetic_y)))])
    positive_mass = weights[y > 0].sum()
    negative_mass = weights[y == 0].sum()
    if not positive_mass or not negative_mass:
        return AbstainingClassifier(), {'reason': 'one_or_both_fitting_classes_absent', 'examples': len(y)}
    weights *= np.where(y > 0, .5 / positive_mass, .5 / negative_mass) * len(y)
    model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=.08,
                                           l2_regularization=1., early_stopping=False, random_state=seed)
    model.fit(x, y, sample_weight=weights)
    return model, {'examples': len(y), 'positives': int(y.sum()),
                   'real_examples': len(real_y), 'real_positives': int(real_y.sum()),
                   'generated_examples': 0 if synthetic_y is None else len(synthetic_y)}


def train(domain, seed, budgets=(16, 64, 256)):
    start = time.monotonic()
    cache = ROOT / 'revision/data/annotation_v1' / domain / f'seed{seed}'
    manifest = json.loads((cache / 'manifest.json').read_text())
    destination = ROOT / 'revision/reports' / f'budget_training_{domain}_seed{seed}_v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    paths = ['revision/budget_learning.py', 'revision/annotation.py', 'tgrepair/matching.py',
             'tgrepair/entity_features.py', 'tgrepair/action_features.py']
    report = {'status': 'running_revision_budget_training', 'domain': domain, 'seed': seed,
              'annotation_manifest_sha256': sha256(cache / 'manifest.json'),
              'implementation': {p: sha256(ROOT / p) for p in paths}, 'models': [],
              'fixed_parameters': {'max_iter': 200, 'max_leaf_nodes': 15, 'learning_rate': .08,
                                   'l2_regularization': 1., 'early_stopping': False,
                                   'generated_real_mass_before_class_balance': [.5, .5],
                                   'calibration_precision_target': .95, 'calibration_minimum_correct': 5},
              'limits': ['Calibration uses only the budgeted target histories and supplies no transfer guarantee.',
                         'The annotation pool and fresh evaluation roots are disjoint; context resources may overlap.',
                         'Historical research development costs are separate from this new-history adaptation budget.']}
    destination.write_text(json.dumps(report, indent=2) + '\n')
    # Hold the generated collection fixed across label-allocation seeds so that
    # variation in the primary budget curves concerns adaptation labels/faults.
    generated_names = {'mixed_original': 'edit_train_v2', 'mixed_varied': 'edit_train_attributes_v3'}
    with threadpool_limits(limits=4):
        for regime in ['real_only', 'mixed_original', 'mixed_varied']:
            generated = None
            if regime in generated_names:
                name = generated_names[regime]
                generated = generated_examples(name)
                report.setdefault('generated_manifests', {})[name] = sha256(ROOT / 'data/edit_learning' / name / 'manifest.json')
            for budget in budgets:
                train_path = cache / f'fit_{budget}.npz'
                valid_path = cache / f'calibration_{budget}.npz'
                expected = manifest['annotation_budgets'][str(budget)]
                if sha256(train_path) != expected['fit']['sha256'] or sha256(valid_path) != expected['calibration']['sha256']:
                    raise ValueError('Annotation cache changed')
                real = np.load(train_path)
                valid = np.load(valid_path)
                models, metrics = [], []
                prediction = np.zeros(len(valid['y']))
                for op in range(3):
                    rm = real['operation'] == op
                    sx = sy = None
                    if generated is not None:
                        gx, gy, gop = generated
                        gm = gop == op
                        sx, sy = gx[gm], gy[gm]
                    model, counts = fit_operation(real['x'][rm], real['y'][rm], sx, sy, seed)
                    models.append(model)
                    vm = valid['operation'] == op
                    if vm.any():
                        prediction[vm] = model.predict_proba(valid['x'][vm])[:, 1]
                    counts['calibration_examples'] = int(vm.sum())
                    counts['calibration_positives'] = int(valid['y'][vm].sum())
                    counts['calibration_ap'] = (float(average_precision_score(valid['y'][vm], prediction[vm]))
                                               if valid['y'][vm].sum() else None)
                    metrics.append(counts)
                thresholds, calibration = select_thresholds(valid['y'], prediction, valid['operation'])
                name = f'budget_{domain}_seed{seed}_{regime}_{budget}_v1'
                model_path = ROOT / 'revision/checkpoints' / (name + '.joblib')
                scores_path = ROOT / 'revision/checkpoints' / (name + '_calibration.npz')
                joblib.dump(models, model_path)
                np.savez_compressed(scores_path, probability=prediction, label=valid['y'], operation=valid['operation'])
                row = {'name': name, 'regime': regime, 'total_verified_root_bundles': budget,
                       'fit_verified_root_bundles': expected['fit']['verified_root_bundles'],
                       'calibration_verified_root_bundles': expected['calibration']['verified_root_bundles'],
                       'annotation_cost': expected, 'operation_metrics': metrics,
                       'model_sha256': sha256(model_path), 'calibration_sha256': sha256(scores_path),
                       'thresholds': thresholds, 'calibration': calibration, 'seconds': time.monotonic() - start}
                report['models'].append(row)
                destination.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps({k: row[k] for k in ['name', 'thresholds', 'seconds']}), flush=True)
    report['status'] = 'completed_revision_budget_training'
    report['seconds'] = time.monotonic() - start
    for path, digest in report['implementation'].items():
        if sha256(ROOT / path) != digest:
            raise RuntimeError('Training implementation changed during execution: ' + path)
    destination.write_text(json.dumps(report, indent=2) + '\n')

