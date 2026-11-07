"""Reproducible matched repeated-corruption study; all outputs stay in this folder."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from revision.annotation import allocate
from revision.budget_learning import fit_operation, select_thresholds
from revision.evaluation import damage_counts
from tgrepair.data import Graph, load_graph, save_graph, sha256
from tgrepair.metrics import traceability, identity_pair_scores
from .method import (MODES, actions, ambiguous_graph, corrupt, decode,
                     score, training_labels, workflow)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIT_VARIANTS = ['mixed', 'independent', 'mild', 'severe', 'missing', 'wrong', 'fragment', 'pipeline']
CAL_VARIANTS = ['mixed', 'independent', 'pipeline']


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def implementation():
    paths = [HERE / 'method.py', HERE / 'run.py', ROOT / 'revision/budget_learning.py',
             ROOT / 'revision/annotation.py', ROOT / 'revision/evaluation.py',
             ROOT / 'revision/transactional_decoding.py']
    paths += list((ROOT / 'tgrepair').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def freeze():
    destination = HERE / 'protocol.json'
    if destination.exists():
        verify(); return
    inputs = {}
    for domain in ['bpi2017', 'bpi2019']:
        for partition in ['annotation', 'confirmation']:
            for name in ['reference_graph.json.gz', 'bundles.json']:
                p = ROOT / 'revision/data/cohorts_v1' / domain / partition / name
                inputs[str(p.relative_to(ROOT))] = sha256(p)
    write(destination, {
        'status': 'frozen_before_followup_model_outcomes', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'design': 'Exploratory follow-up on previously evaluated source cohorts; not independent confirmation.',
        'domains': ['bpi2017', 'bpi2019'], 'seeds': [0, 1, 2], 'budgets': [16, 64, 256],
        'fit_fraction': .75, 'arms': ['real8', 'real8_generated'],
        'fit_corruptions': FIT_VARIANTS, 'calibration_corruptions': CAL_VARIANTS,
        'calibration_precision_targets': [.95, .99], 'minimum_correct': 5,
        'fit_event_queries_per_corruption': 256, 'calibration_event_queries_per_corruption': 512,
        'generated_worlds': 32, 'generated_roots_per_world': 16,
        'generated_ambiguity_graphs': 2,
        'fit_settings': {'iterations': 200, 'leaves': 15, 'learning_rate': .08, 'l2': 1.,
                         'real_generated_weight_mass': [.5, .5], 'class_balance': True},
        'test_roots': 128, 'test_selection': 'first 128 hashes of improvement_v2_test:<opaque root>, same for all seeds',
        'primary': 'mixed faults, both arms, all budgets/seeds, both precision targets',
        'ablations': 'generated arm, budget64, all seeds, both precision targets; project before retrieval',
        'ablation_modes': list(MODES),
        'stress': 'clean and pipeline, both arms, budget64, all seeds, precision .95',
        'safety': 'with/without guards on mixed, both arms, budget64, all seeds, precision .99',
        'decoding': 'one pass, atomic exactly-one-root substitution, commuting edits',
        'safety_guards': {'merge_score_floor': .99, 'mutual_best_margin': .05,
                          'reject_cooccurring_merges': True, 'reject_tied_additions': True},
        'interpretation': 'Compare entire fixed recovery/damage grid, not a test-selected deployment threshold. No guaranteed harm bound.',
        'human_evidence': 'Natural-error labels and human verification time unavailable; user authorized skipping these.',
        'implementation': implementation(), 'inputs': inputs,
    })


def verify():
    p = json.loads((HERE / 'protocol.json').read_text())
    for group in ['implementation', 'inputs']:
        for name, digest in p[group].items():
            if sha256(ROOT / name) != digest:
                raise ValueError('Frozen input changed: ' + name)
    return p


def subset(graph, bundles, roots):
    targets = set().union(*(set(bundles[r]) for r in roots))
    events = [e for e in graph.events if set(e.participants) & targets]
    used = targets | {o for e in events for o in e.participants}
    # A bundle cannot expose another target history through a shared event.
    all_targets = set().union(*(set(v) for v in bundles.values()))
    assert used & all_targets == targets
    result = Graph(graph.name + '_subset', {o: graph.objects[o] for o in sorted(used)}, events)
    result.validate()
    return result


def concatenate(rows):
    return {key: np.concatenate([r[key] for r in rows]) for key in ('x', 'y', 'operation')}


def example_cache(path, graphs, types, mode, event_limit, seed, ambiguous=False):
    meta_path = path.with_suffix('.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        assert sha256(path) == meta['sha256']
        return dict(np.load(path)), meta
    start = time.perf_counter(); cpu = time.process_time(); rows = []
    sources = []
    for i, (graph, reference) in enumerate(graphs):
        observation = actions(graph, types, mode, event_limit, seed + i)
        labels = np.zeros(len(observation.edits), dtype=np.float32) if reference is None else training_labels(observation, reference)
        rows.append({'x': observation.action_features, 'y': labels, 'operation': observation.operations})
        sources.append({'events': len(graph.events), 'objects': len(graph.objects),
                        'action_rows': len(labels), 'ambiguous': reference is None})
    result = concatenate(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    meta = {'sha256': sha256(path), 'seconds': time.perf_counter() - start,
            'cpu_seconds': time.process_time() - cpu, 'mode': mode, 'sources': sources,
            'actions': len(result['y']), 'positives': [int(result['y'][result['operation'] == op].sum()) for op in range(3)]}
    write(meta_path, meta)
    print(json.dumps({'cache': str(path.relative_to(HERE)), 'actions': meta['actions'], 'seconds': meta['seconds']}), flush=True)
    return result, meta


def prepare_generated():
    verify()
    graphs, counts = [], []
    for i in range(32):
        clean, count = workflow(341000 + i)
        graph, reference = corrupt(clean, {'Root', 'Item'}, 342000 + i, FIT_VARIANTS[i % 8])
        folder = HERE / 'data/generated' / f'world{i}'
        save_graph(clean, folder / 'clean.json.gz'); save_graph(graph, folder / 'observed.json.gz')
        write(folder / 'reference.json', reference)
        counts.append({'world': i, 'events': len(clean.events), 'activities': count,
                       'fault': FIT_VARIANTS[i % 8], 'clean_sha256': sha256(folder / 'clean.json.gz'),
                       'observed_sha256': sha256(folder / 'observed.json.gz'),
                       'reference_sha256': sha256(folder / 'reference.json')})
        graphs.append((graph, reference))
    graphs.extend((ambiguous_graph(343000 + i), None) for i in range(2))
    for mode in MODES:
        example_cache(HERE / 'data' / f'generated_{mode}.npz', graphs, {'Root', 'Item'}, mode, 256, 344000)
    write(HERE / 'generated_manifest.json', {'worlds': counts, 'ambiguity_graphs': 2,
                                           'protocol_sha256': sha256(HERE / 'protocol.json')})


def train(real, valid, generated, seed, path):
    meta_path = path.with_suffix('.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text()); assert sha256(path) == meta['sha256']
        return joblib.load(path), meta
    start = time.perf_counter(); cpu = time.process_time()
    models, counts = [], []
    predictions = np.zeros(len(valid['y']))
    for op in range(3):
        rm = real['operation'] == op
        gx = gy = None
        if generated is not None:
            gm = generated['operation'] == op
            gx, gy = generated['x'][gm], generated['y'][gm]
        model, count = fit_operation(real['x'][rm], real['y'][rm], gx, gy, seed)
        models.append(model); counts.append(count)
        vm = valid['operation'] == op
        if vm.any():
            predictions[vm] = model.predict_proba(valid['x'][vm])[:, 1]
    thresholds = {}
    for precision in [.95, .99]:
        values, details = select_thresholds(valid['y'], predictions, valid['operation'], precision)
        thresholds[str(precision)] = {'values': values, 'details': details}
    path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(models, path)
    np.savez_compressed(path.with_suffix('.calibration.npz'), probability=predictions,
                        y=valid['y'], operation=valid['operation'])
    meta = {'sha256': sha256(path), 'seconds': time.perf_counter() - start,
            'cpu_seconds': time.process_time() - cpu, 'counts': counts, 'thresholds': thresholds}
    write(meta_path, meta)
    print(json.dumps({'model': path.stem, 'seconds': meta['seconds'], 'thresholds': {p: t['values'] for p, t in thresholds.items()}}), flush=True)
    return models, meta


def evaluate(observation, reference, types, root_type, models, model_meta, precision, destination,
             safe=True, feature_seconds=0., feature_cpu_seconds=0.):
    result_path = destination / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    start = time.perf_counter(); cpu = time.process_time()
    scores = score(models, observation)
    state, audit = decode(observation, scores, model_meta['thresholds'][str(precision)]['values'], root_type, safe)
    result = {'history': traceability(state.graph, reference, types, state.clusters),
              'identity': identity_pair_scores(state.graph, reference, types, state.clusters),
              'events': damage_counts(observation.graph, state.graph, reference, types, state.clusters),
              'seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu,
              'feature_seconds': feature_seconds, 'feature_cpu_seconds': feature_cpu_seconds,
              'candidate_actions': len(scores), 'edits': len(audit['edits']),
              'precision_target': precision, 'safe': safe, 'model_sha256': model_meta['sha256']}
    save_graph(state.graph, destination / 'repaired.json.gz')
    write(destination / 'clusters.json', state.clusters); write(destination / 'audit.json', audit)
    # Full reversible audit, including events affected indirectly by a merge.
    after_events = state.graph.events
    while state.log:
        state.undo()
    assert state.graph.events == observation.graph.events
    assert all(o == c for o, c in state.clusters.items())
    result['undo_verified'] = True
    result['artifacts'] = {name: sha256(destination / name) for name in ['repaired.json.gz', 'clusters.json', 'audit.json']}
    write(result_path, result)
    return result


def run_domain(domain):
    verify(); start = time.perf_counter(); cpu = time.process_time()
    output = HERE / 'results' / domain
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'revision/data/cohorts_v1' / domain
    manifest = json.loads((ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json').read_text())
    types = set(manifest['target_types']); root_type = manifest['root_type']
    pool = load_graph(source / 'annotation/reference_graph.json.gz')
    bundles = json.loads((source / 'annotation/bundles.json').read_text())
    test_source = load_graph(source / 'confirmation/reference_graph.json.gz')
    test_bundles = json.loads((source / 'confirmation/bundles.json').read_text())
    test_roots = sorted(test_bundles, key=lambda r: hashlib.sha256(('improvement_v2_test:' + r).encode()).hexdigest())[:128]
    clean_test = subset(test_source, test_bundles, test_roots)
    summary = {'domain': domain, 'protocol_sha256': sha256(HERE / 'protocol.json'),
               'test_roots': test_roots, 'test_summary': clean_test.summary(), 'allocations': [], 'evaluations': []}
    generated = {}
    with threadpool_limits(limits=4):
        for seed in range(3):
            allocations = allocate(bundles, 351000 + seed)
            test = {}
            for variant in ['mixed', 'clean', 'pipeline']:
                graph, reference = corrupt(clean_test, types, 352000 + seed, variant)
                folder = HERE / 'data' / domain / f'test_seed{seed}_{variant}'
                save_graph(graph, folder / 'observed.json.gz'); write(folder / 'reference.json', reference)
                test[variant] = graph, reference
            observations = {}
            for budget in [16, 64, 256]:
                split = allocations[budget]
                assert split['fit'].isdisjoint(split['calibration'])
                role_graphs, costs = {}, {}
                for role in ['fit', 'calibration']:
                    clean = subset(pool, bundles, split[role])
                    costs[role] = {'root_ids': sorted(split[role]), 'roots': len(split[role]),
                                   'events': len(clean.events), 'target_entities': sum(t in types for t in clean.objects.values())}
                    variants = FIT_VARIANTS if role == 'fit' else CAL_VARIANTS
                    role_graphs[role] = [corrupt(clean, types, 353000 + 100 * seed + i + (50 if role == 'calibration' else 0), variant)
                                         for i, variant in enumerate(variants)]
                summary['allocations'].append({'seed': seed, 'budget': budget, **costs})
                for mode in MODES if budget == 64 else ['full']:
                    folder = HERE / 'data' / domain / f'seed{seed}_budget{budget}_{mode}'
                    real, real_meta = example_cache(folder / 'fit.npz', role_graphs['fit'], types, mode, 256, 354000 + seed)
                    valid, valid_meta = example_cache(folder / 'calibration.npz', role_graphs['calibration'], types, mode, 512, 355000 + seed)
                    if mode not in generated:
                        generated[mode] = dict(np.load(HERE / 'data' / f'generated_{mode}.npz'))
                    arms = ['real8', 'real8_generated'] if mode == 'full' else ['real8_generated']
                    for arm in arms:
                        model_path = HERE / 'models' / domain / f'seed{seed}_budget{budget}_{mode}_{arm}.joblib'
                        models, model_meta = train(real, valid, generated[mode] if arm.endswith('generated') else None, seed, model_path)
                        conditions = [('mixed', p, True) for p in [.95, .99]]
                        if budget == 64 and mode == 'full':
                            conditions += [('clean', .95, True), ('pipeline', .95, True), ('mixed', .99, False)]
                        for variant, precision, safe in conditions:
                            key = variant, mode
                            if key not in observations:
                                t = time.perf_counter(); c = time.process_time()
                                obs = actions(test[variant][0], types, mode)
                                observations[key] = obs, time.perf_counter() - t, time.process_time() - c
                            obs, seconds, cpu_seconds = observations[key]
                            destination = output / f's{seed}_b{budget}_{mode}_{arm}_{variant}_p{precision}_safe{int(safe)}'
                            result = evaluate(obs, test[variant][1], types, root_type, models, model_meta,
                                              precision, destination, safe, seconds, cpu_seconds)
                            row = {'seed': seed, 'budget': budget, 'mode': mode, 'arm': arm,
                                   'variant': variant, 'precision': precision, 'safe': safe,
                                   'result': str((destination / 'result.json').relative_to(HERE)),
                                   'result_sha256': sha256(destination / 'result.json'),
                                   'model': str(model_path.relative_to(HERE)), 'model_meta': model_meta,
                                   'fit_cache_sha256': real_meta['sha256'], 'calibration_cache_sha256': valid_meta['sha256'],
                                   'f1': result['history']['f1'], 'events': result['events']}
                            summary['evaluations'].append(row)
                            write(output / 'progress.json', summary)
                            print(json.dumps({k: v for k, v in row.items() if k not in ['model_meta', 'events']}, default=str), flush=True)
                        if budget == 64 and mode == 'full':
                            ambiguity = actions(ambiguous_graph(356000 + seed), {'Root'}, mode)
                            ambiguity_scores = score(models, ambiguity)
                            for safe in [False, True]:
                                state, audit = decode(ambiguity, ambiguity_scores, model_meta['thresholds']['0.95']['values'], 'Root', safe)
                                write(output / f'ambiguity_s{seed}_{arm}_safe{int(safe)}.json',
                                      {'events': len(ambiguity.graph.events), 'unresolved': sum(not e.participants for e in state.graph.events),
                                       'unsupported_edits': len(audit['edits']), 'audit': audit})
                    del real, valid
                write(output / 'progress.json', summary)
    verify()
    summary.update({'status': 'completed', 'seconds': time.perf_counter() - start,
                    'cpu_seconds': time.process_time() - cpu, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'hardware': platform.platform(), 'logical_cpus': os.cpu_count(), 'threads_per_worker': 4})
    write(output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'generated', 'domain'])
    parser.add_argument('--domain', choices=['bpi2017', 'bpi2019'])
    args = parser.parse_args()
    if args.stage == 'freeze': freeze()
    elif args.stage == 'generated': prepare_generated()
    else:
        if args.domain is None: raise ValueError('Domain required')
        run_domain(args.domain)


if __name__ == '__main__':
    main()
