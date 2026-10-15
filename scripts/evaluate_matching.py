"""Apply an already fitted local matcher to a different diagnostic pool."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_baseline_probe import evaluate_cache
from tgrepair.data import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe', required=True)
    p.add_argument('--model', default='matching_v1')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    destination = ROOT / 'reports' / (args.output + '.json')
    if destination.exists():
        raise FileExistsError(destination)
    source = ROOT / 'reports' / (args.probe + '.json')
    checkpoint = ROOT / 'checkpoints' / (args.model + '.joblib')
    # Checkpoints here were created locally by fit_matching_baseline.py.
    model = joblib.load(checkpoint)
    probe = json.loads(source.read_text())
    runs = []
    with threadpool_limits(limits=8):
        for r in probe['runs']:
            result = {'dataset': r['dataset'], 'mode': r['mode'], 'seed': r['seed'],
                      'training_queries': r['dataset'] == 'bpi2012_fit',
                      **evaluate_cache(dict(np.load(ROOT / r['cache'])), model)}
            runs.append(result)
            print(json.dumps({k: result[k] for k in ['dataset','mode','candidate_recall_macro','recall_at_known_cardinality']}), flush=True)
    destination.write_text(json.dumps({'status': 'development_matching_evaluation',
                                      'source_report': str(source.relative_to(ROOT)), 'source_hash': sha256(source),
                                      'model': str(checkpoint.relative_to(ROOT)), 'model_hash': sha256(checkpoint),
                                      'runs': runs}, indent=2) + '\n')


if __name__ == '__main__':
    main()
