"""Fit a strong feature-based matcher on development-fit queries only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_baseline_probe import evaluate_cache
from tgrepair.data import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', default='baseline_probe_v1')
    parser.add_argument('--output', default='matching_v1')
    args = parser.parse_args()
    source = ROOT / 'reports' / (args.probe + '.json')
    report_path = ROOT / 'reports' / (args.output + '.json')
    if report_path.exists():
        raise FileExistsError(report_path)
    probe = json.loads(source.read_text())
    train = [r for r in probe['runs'] if r['dataset'] in {'bpi2012_fit','bpi2012_fit_attributes'}]
    assert train
    arrays = [dict(np.load(ROOT / r['cache'])) for r in train]
    x, y = np.concatenate([c['x'] for c in arrays]), np.concatenate([c['y'] for c in arrays])
    weights = np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=0.08,
                                           l2_regularization=1.0, random_state=0, early_stopping=False)
    start = time.monotonic()
    with threadpool_limits(limits=8):
        model.fit(x, y, sample_weight=weights)
        results = []
        for r in probe['runs']:
            cache = dict(np.load(ROOT / r['cache']))
            results.append({'dataset': r['dataset'], 'mode': r['mode'], 'seed': r['seed'],
                            'training_queries': r['dataset'] in {'bpi2012_fit','bpi2012_fit_attributes'},
                            **evaluate_cache(cache, model)})
    (ROOT / 'checkpoints').mkdir(exist_ok=True)
    model_path = ROOT / 'checkpoints' / (args.output + '.joblib')
    joblib.dump(model, model_path)
    report = {'status': 'development_matching_baseline', 'source_report': str(source.relative_to(ROOT)),
              'source_hash': sha256(source), 'training': 'BPI2012 development-fit only',
              'rows': len(y), 'positive_rows': int(y.sum()), 'parameters': model.get_params(),
              'model_sha256': sha256(model_path), 'seconds': time.monotonic() - start,
              'runs': results}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    for r in results:
        print(json.dumps({k: r[k] for k in ['dataset', 'mode', 'seed', 'training_queries', 'recall_at_known_cardinality']}))


if __name__ == '__main__':
    main()
