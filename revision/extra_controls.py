"""Zero-target-label, calibration-only, and root-transaction controls."""
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from revision.budget_learning import select_thresholds
from revision.evaluation import development_task, damage_counts
from revision.transactional_decoding import decode_atomic_root
from tgrepair.data import load_graph, save_graph, sha256
from tgrepair.decoding import FeatureScorer, decode
from tgrepair.metrics import traceability, identity_pair_scores
from tgrepair.application_queries import audit_queries

ROOT = Path(__file__).resolve().parents[1]


def prepare_models(domain, seed):
    destination = ROOT / 'revision/reports' / f'extra_models_{domain}_seed{seed}_v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    cache = ROOT / 'revision/data/annotation_v1' / domain / f'seed{seed}'
    manifest = json.loads((cache / 'manifest.json').read_text())
    report = {'status': 'preparing_extra_controls', 'domain': domain, 'seed': seed, 'models': [],
              'implementation_sha256': sha256(__file__),
              'annotation_manifest_sha256': sha256(cache / 'manifest.json'),
              'contract': 'Budgets count incremental target verification. Zero means no new target labels; historical research costs are not zero. Calibration-only controls use the same complete verified bundles as fitting-plus-calibration controls.'}
    with threadpool_limits(limits=4):
        for variant, model_name, calibration_name in [
            ('original', 'edit_feature_v2', 'edit_feature_v2_attributes_calibration_v3'),
            ('varied', 'edit_feature_attributes_v3', 'edit_feature_attributes_v3')]:
            model_path = ROOT / 'checkpoints' / (model_name + '.joblib')
            calibration_path = ROOT / 'data/edit_learning' / (calibration_name + '_validation_scores.npz')
            generated = np.load(calibration_path)
            thresholds, details = select_thresholds(generated['label'], generated['probability'], generated['operation'])
            report['models'].append({'name': f'extra_{domain}_seed{seed}_generated_{variant}_0_v1',
                'regime': 'generated_' + variant, 'total_verified_root_bundles': 0,
                'model_path': str(model_path.relative_to(ROOT)), 'model_sha256': sha256(model_path),
                'thresholds': thresholds, 'calibration': details,
                'calibration_source': 'generated_varied_observations', 'calibration_sha256': sha256(calibration_path)})
            if variant != 'varied':
                continue
            models = joblib.load(model_path)
            for budget in [16, 64, 256]:
                xs, ys, ops = [], [], []
                for role in ['fit', 'calibration']:
                    path = cache / f'{role}_{budget}.npz'
                    if sha256(path) != manifest['annotation_budgets'][str(budget)][role]['sha256']:
                        raise ValueError('Annotation cache changed')
                    data = np.load(path)
                    xs.append(data['x']); ys.append(data['y']); ops.append(data['operation'])
                x, y, operation = np.concatenate(xs), np.concatenate(ys), np.concatenate(ops)
                probability = np.zeros(len(y))
                for op, model in enumerate(models):
                    mask = operation == op
                    if mask.any():
                        probability[mask] = model.predict_proba(x[mask])[:, 1]
                thresholds, details = select_thresholds(y, probability, operation)
                name = f'extra_{domain}_seed{seed}_calibration_only_varied_{budget}_v1'
                scores = ROOT / 'revision/checkpoints' / (name + '_calibration.npz')
                np.savez_compressed(scores, probability=probability, label=y, operation=operation)
                report['models'].append({'name': name, 'regime': 'calibration_only_varied',
                    'total_verified_root_bundles': budget, 'fit_verified_root_bundles': 0,
                    'calibration_verified_root_bundles': budget,
                    'annotation_cost': manifest['annotation_budgets'][str(budget)],
                    'model_path': str(model_path.relative_to(ROOT)), 'model_sha256': sha256(model_path),
                    'thresholds': thresholds, 'calibration': details,
                    'calibration_source': 'all_budgeted_target_bundles_no_model_refitting',
                    'calibration_sha256': sha256(scores)})
    report['status'] = 'completed_extra_model_preparation'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(destination), 'models': len(report['models'])}), flush=True)


def evaluate(domain, seed, suite):
    if suite not in {'extra', 'reliability'}:
        raise ValueError(suite)
    directory, task = development_task(domain, seed, 'mixed')
    graph = load_graph(directory / 'observed.json.gz')
    reference = json.loads((directory / 'reference.json').read_text())
    types = set(task['target_types'])
    cohort = json.loads((ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json').read_text())
    if suite == 'extra':
        specification = ROOT / 'revision/reports' / f'extra_models_{domain}_seed{seed}_v1.json'
        spec = json.loads(specification.read_text())
        jobs = [dict(row, decoder='original') for row in spec['models']]
    else:
        specification = ROOT / 'revision/reports' / f'budget_training_{domain}_seed{seed}_v1.json'
        spec = json.loads(specification.read_text())
        jobs = []
        # Fix the middle budget and two representative regimes before outcomes.
        # A stricter original decoder is the essential reliability comparator.
        for model in spec['models']:
            if model['total_verified_root_bundles'] != 64 or model['regime'] not in {'real_only', 'mixed_varied'}:
                continue
            model_path = 'revision/checkpoints/' + model['name'] + '.joblib'
            jobs.append(dict(model, name=model['name'] + '_atomic_root', decoder='atomic_root', model_path=model_path))
            calibration = np.load(ROOT / 'revision/checkpoints' / (model['name'] + '_calibration.npz'))
            thresholds, details = select_thresholds(calibration['label'], calibration['probability'],
                                                      calibration['operation'], precision_target=.99)
            jobs.append(dict(model, name=model['name'] + '_strict99', decoder='original',
                             model_path=model_path, thresholds=thresholds, calibration=details))
    report_path = ROOT / 'revision/reports' / f'development_{suite}_{domain}_seed{seed}_v1.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    report = {'status': 'running_development_controls', 'suite': suite, 'task': task,
              'specification_sha256': sha256(specification), 'implementation_sha256': sha256(__file__),
              'transaction_implementation_sha256': sha256(ROOT / 'revision/transactional_decoding.py'),
              'before': traceability(graph, reference, types), 'results': [],
              'declared_jobs': [{k: r[k] for k in ['name', 'decoder', 'thresholds', 'model_sha256']} for r in jobs],
              'root_contract': {'type': cohort['root_type'], 'minimum': 1, 'maximum': 1,
                                'source': 'Public source representation: each retained event has one application/order root. Applied only by atomic_root control; no child multiplicity rule.'}}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    start = time.monotonic()
    with threadpool_limits(limits=4):
        for row in jobs:
            model_path = ROOT / row['model_path']
            if sha256(model_path) != row['model_sha256']:
                raise ValueError('Model checksum changed')
            kwargs = dict(rounds=4, participant_budget=128, pair_budget=64,
                          progress=lambda r: print(json.dumps({'model': row['name'], **r}), flush=True))
            scorer = FeatureScorer(model_path)
            if row['decoder'] == 'atomic_root':
                state, audit = decode_atomic_root(graph, types, scorer, row['thresholds'], cohort['root_type'], **kwargs)
            else:
                state, audit = decode(graph, types, scorer, row['thresholds'], **kwargs)
            result = dict(row, history=traceability(state.graph, reference, types, state.clusters),
                identity=identity_pair_scores(state.graph, reference, types, state.clusters),
                events=damage_counts(graph, state.graph, reference, types, state.clusters),
                queries=audit_queries(state.graph, reference, types, state.clusters),
                seconds=audit['seconds'], edits=audit['edit_count'])
            out = ROOT / 'revision/data/decoding_v1' / f'development_{suite}_{domain}_seed{seed}' / row['name']
            out.mkdir(parents=True)
            save_graph(state.graph, out / 'repaired.json.gz')
            (out / 'clusters.json').write_text(json.dumps(state.clusters) + '\n')
            (out / 'audit.json').write_text(json.dumps(audit) + '\n')
            result['audit_sha256'] = sha256(out / 'audit.json')
            report['results'].append(result)
            report['seconds'] = time.monotonic() - start
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'model': row['name'], 'f1': result['history']['f1'],
                              'events': result['events'], 'seconds': result['seconds']}), flush=True)
    report['status'] = 'completed_development_controls'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
