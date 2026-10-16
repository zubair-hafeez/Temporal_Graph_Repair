"""Scoped entry point for this paper's revision experiments."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task', required=True)
    sub.add_parser('runtime')
    cohorts = sub.add_parser('prepare-cohorts')
    cohorts.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    native_cohorts = sub.add_parser('prepare-native-cohort')
    native_cohorts.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    annotation = sub.add_parser('prepare-annotation')
    annotation.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    annotation.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    budget = sub.add_parser('train-budget')
    budget.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    budget.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    evaluation = sub.add_parser('evaluate-budget-development')
    evaluation.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    evaluation.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    evaluation.add_argument('--variant', choices=['mixed', 'clean', 'pipeline'], default='mixed')
    here = sub.add_parser('here-development')
    here.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    here.add_argument('--seed', type=int, default=0)
    here.add_argument('--missing-rate', type=float, choices=[.1, .3, .5], default=.1)
    here.add_argument('--epochs', type=int, choices=[1, 2, 30], default=2)
    here.add_argument('--max-roots', type=int, choices=[128, 256, 512], default=128)
    here.add_argument('--device', choices=['cuda:0', 'cuda:1', 'cpu'], default='cuda:0')
    controls = sub.add_parser('native-controls-development')
    controls.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    controls.add_argument('--seed', type=int, default=0)
    controls.add_argument('--missing-rate', type=float, choices=[.1, .3, .5], default=.1)
    controls.add_argument('--max-roots', type=int, choices=[128, 256, 512], default=512)
    extras = sub.add_parser('prepare-extra-controls')
    extras.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    extras.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    evaluate_controls = sub.add_parser('evaluate-controls-development')
    evaluate_controls.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    evaluate_controls.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    evaluate_controls.add_argument('--suite', choices=['extra', 'reliability'], required=True)
    native = sub.add_parser('native-confirmation')
    native.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    native.add_argument('--device', choices=['cuda:0', 'cuda:1'], required=True)
    args = parser.parse_args()
    if args.task == 'runtime':
        import torch
        report = {'python': sys.version, 'torch': torch.__version__,
                  'cuda_available': torch.cuda.is_available(), 'devices': []}
        if report['cuda_available']:
            for index in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{index}')
                value = torch.ones(8, device=device).sum().item()
                report['devices'].append({'index': index, 'name': torch.cuda.get_device_name(index),
                                          'smoke_sum': value})
        print(json.dumps(report), flush=True)
        if not report['cuda_available']:
            raise RuntimeError('GPU training required for native published Transformer comparison; CUDA unavailable')
    elif args.task == 'prepare-cohorts':
        from revision.cohorts import prepare
        prepare(args.domain)
    elif args.task == 'prepare-annotation':
        from revision.annotation import prepare
        prepare(args.domain, args.seed)
    elif args.task == 'prepare-native-cohort':
        from revision.native_cohort import prepare
        prepare(args.domain)
    elif args.task == 'train-budget':
        from revision.budget_learning import train
        train(args.domain, args.seed)
    elif args.task == 'evaluate-budget-development':
        from revision.evaluation import evaluate_development
        evaluate_development(args.domain, args.seed, args.variant)
    elif args.task == 'here-development':
        from revision.here_benchmark import train_development
        train_development(args.domain, args.seed, args.missing_rate, args.epochs, args.max_roots, args.device)
    elif args.task == 'native-controls-development':
        from revision.native_controls import evaluate
        evaluate(args.domain, args.seed, args.missing_rate, args.max_roots)
    elif args.task == 'prepare-extra-controls':
        from revision.extra_controls import prepare_models
        prepare_models(args.domain, args.seed)
    elif args.task == 'evaluate-controls-development':
        from revision.extra_controls import evaluate
        evaluate(args.domain, args.seed, args.suite)
    elif args.task == 'native-confirmation':
        from revision.native_protocol import run_suite
        run_suite(args.domain, args.device)


if __name__ == '__main__':
    main()
