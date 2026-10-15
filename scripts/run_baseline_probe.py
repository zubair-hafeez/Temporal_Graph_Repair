"""Development known-hole experiment, never a final paper evaluation."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import load_graph, sha256
from tgrepair.corruption import remove_participation
from tgrepair.matching import FEATURE_NAMES, ObservedIndex, expected_topk_recall, fixed_scores


def evaluate_cache(cache, learned=None):
    x, y, offsets, counts = cache['x'], cache['y'], cache['offsets'], cache['counts']
    scores = fixed_scores(x)
    if learned is not None:
        scores['learned_matching'] = learned.predict_proba(x)[:, 1]
    hits = {name: [] for name in scores}
    hits['uniform_candidates'] = []
    recall = []
    for i, k in enumerate(counts):
        start, end = offsets[i:i + 2]
        pos = y[start:end]
        recall.append(float(pos.sum()) / k)
        for name, values in scores.items():
            hits[name].append(expected_topk_recall(values[start:end], pos, int(k)))
        hits['uniform_candidates'].append(float(pos.sum()) / max(1, end - start)
                                           * min(k, end - start) / k)
    return {
        'queries': len(counts), 'target_links': int(counts.sum()),
        'candidate_recall_macro': float(np.mean(recall)),
        'candidate_recall_micro': float(y.sum() / counts.sum()),
        'multi_target_queries': int((counts > 1).sum()),
        'recall_at_known_cardinality': {k: float(np.mean(v)) for k, v in hits.items()},
        'per_type': {typ: {
            'queries': int(np.sum(cache['types'] == typ)),
            'candidate_recall_macro': float(np.mean(np.asarray(recall)[cache['types'] == typ])),
            'recall_at_known_cardinality': {k: float(np.mean(np.asarray(v)[cache['types'] == typ]))
                                          for k, v in hits.items()}}
            for typ in sorted(set(cache['types']))},
    }


def build_cache(graph, seed, mode, budget, max_queries):
    eligible = set(graph.objects.values()) - {'Resource', 'Forklift', 'Truck', 'Vehicle'}
    observed, holes = remove_participation(graph, seed, mode, eligible, 0.2)
    index = ObservedIndex(observed)
    labels = defaultdict(set)
    for h in holes:
        labels[h.event_uid, h.object_type].add(h.reference_object)
    queries = sorted(labels)
    rng = random.Random(seed + 94721)
    rng.shuffle(queries)
    queries = queries[:max_queries]
    features, positives, offsets, counts, types = [], [], [0], [], []
    counters = Counter()
    for eid, typ in queries:
        targets = labels[eid, typ]
        candidates = index.candidates(eid, typ, budget)
        features.append(index.features(eid, candidates))
        positives.extend(o in targets for o in candidates)
        counts.append(len(targets))
        offsets.append(offsets[-1] + len(candidates))
        types.append(typ)
        counters['queries_without_visible_partner'] += not index.events[eid].participants
        counters['truths_without_observed_history'] += sum(o not in index.hist for o in targets)
    cache = {'x': np.concatenate(features), 'y': np.asarray(positives, dtype=bool),
             'offsets': np.asarray(offsets, dtype=np.int64), 'counts': np.asarray(counts, dtype=np.int64),
             'types': np.asarray(types)}
    info = {'observed_graph': observed.summary(), 'eligible_types': sorted(eligible),
            'all_hidden_links': len(holes), 'all_query_groups': len(labels),
            'severity': len(holes) / sum(graph.objects[o] in eligible for e in graph.events for o in e.participants),
            'candidate_budget': budget, 'time_scale_seconds': index.scale, **dict(counters)}
    return cache, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['bpi2012_fit', 'bpi2012_tune', 'logistics'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--modes', nargs='+', default=['independent', 'contiguous'])
    parser.add_argument('--queries', type=int, default=1000)
    parser.add_argument('--budget', type=int, default=128)
    parser.add_argument('--output', default='baseline_probe_v1')
    args = parser.parse_args()
    assert set(args.datasets) <= {'logistics', 'bpi2012_fit', 'bpi2012_tune', 'bpi2012_check',
                                 'bpi2012_fit_attributes', 'bpi2012_tune_attributes', 'bpi2012_check_attributes'}
    destination = ROOT / 'reports' / (args.output + '.json')
    if destination.exists():
        raise FileExistsError('Preserve previous results; choose a new output name.')
    report = {'status': 'exploratory_known_hole_track', 'config': vars(args),
              'features': FEATURE_NAMES,
              'code_hashes': {str(p.relative_to(ROOT)): sha256(p)
                              for folder in ('tgrepair', 'scripts') for p in sorted((ROOT / folder).glob('*.py'))},
              'limits': ['Missing event/type and cardinality supplied for diagnostic scoring.',
                         'Reference targets do not enter candidate retrieval or features.',
                         'No identity fragmentation or wrong-link diagnosis evaluated.',
                         'No object-object or descriptive attribute evidence in this track.',
                         'All runs are development, not final target outcomes.'], 'runs': []}
    cache_dir = ROOT / 'data/probe_cache' / args.output
    cache_dir.mkdir(parents=True)
    for dataset in args.datasets:
        graph_path = ROOT / 'data/development' / (dataset + '.json.gz')
        graph = load_graph(graph_path)
        for mode in args.modes:
            for seed in args.seeds:
                start = time.monotonic()
                cache, info = build_cache(graph, seed, mode, args.budget, args.queries)
                path = cache_dir / f'{dataset}_{mode}_{seed}.npz'
                np.savez_compressed(path, **cache)
                result = {'dataset': dataset, 'mode': mode, 'seed': seed,
                          'graph_hash': sha256(graph_path), 'cache': str(path.relative_to(ROOT)),
                          **info, **evaluate_cache(cache), 'seconds': time.monotonic() - start}
                report['runs'].append(result)
                destination.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps({k: result[k] for k in ('dataset', 'mode', 'seed', 'queries',
                                                         'candidate_recall_macro', 'recall_at_known_cardinality', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
