"""Audit every saved repair and summarize all prespecified follow-up outcomes."""
from collections import defaultdict
from itertools import combinations
import csv
import json
from pathlib import Path

import numpy as np
import joblib
from threadpoolctl import threadpool_limits

from tgrepair.data import load_graph, sha256
from tgrepair.edits import Edit, RepairState
from .run import HERE, ROOT, verify, write
from revision.budget_learning import select_thresholds


def independent_metrics(original, repaired, reference, types, clusters):
    """Independent set-based history and full cluster-contamination metrics."""
    assert original.objects == repaired.objects
    assert {e.uid for e in original.events} == {e.uid for e in repaired.events}
    assert all(clusters[o] == o for o, typ in original.objects.items() if typ not in types)
    expected, retrieved, aliases = defaultdict(set), defaultdict(set), defaultdict(list)
    identities = reference['identity']
    for uid, ids in reference['event_participation'].items():
        for o in ids:
            if original.objects[o] in types: expected[o].add(uid)
    for o, typ in original.objects.items():
        if typ in types: aliases[identities[o]].append(o)
    for e in repaired.events:
        for o in e.participants:
            if original.objects[o] in types: retrieved[clusters[o]].add(e.uid)
    means, exact = [], []
    for owner, truth in expected.items():
        means.append(np.mean([2 * len(truth & retrieved[clusters[o]]) / (len(truth) + len(retrieved[clusters[o]])) for o in aliases[owner]]))
        exact.append(np.mean([truth == retrieved[clusters[o]] for o in aliases[owner]]))
    groups = defaultdict(set)
    for o, c in clusters.items():
        if original.objects[o] in types: groups[c].add(identities[o])
    before = {e.uid: e for e in original.events}
    damage = recovery = correct_before = incorrect_before = 0
    for e in repaired.events:
        old = before[e.uid]
        assert (e.time, e.kind, e.attributes) == (old.time, old.kind, old.attributes)
        assert {o for o in e.participants if original.objects[o] not in types} == {o for o in old.participants if original.objects[o] not in types}
        truth = {o for o in reference['event_participation'][e.uid] if original.objects[o] in types}
        prior = {identities[o] for o in old.participants if original.objects[o] in types}
        after = {owner for o in e.participants if original.objects[o] in types for owner in groups[clusters[o]]}
        if prior == truth:
            correct_before += 1; damage += after != truth
        else:
            incorrect_before += 1; recovery += after == truth
    cluster_members = defaultdict(list)
    for o, c in clusters.items():
        if original.objects[o] in types: cluster_members[c].append(o)
    pairs = [pair for members in cluster_members.values() for pair in combinations(members, 2)]
    correct_pairs = sum(identities[a] == identities[b] for a, b in pairs)
    return {'f1': float(np.mean(means)), 'exact': float(np.mean(exact)),
            'false_pairs': len(pairs) - correct_pairs, 'correct_pairs': correct_pairs,
            'damage': damage, 'recovery': recovery, 'correct_before': correct_before, 'incorrect_before': incorrect_before}


def main():
    verify(); records = []; checked = {}; cache = {}; verified_models = set()
    for domain in ['bpi2017', 'bpi2019']:
        summary_path = HERE / 'results' / domain / 'summary.json'
        summary = json.loads(summary_path.read_text())
        assert summary['status'] == 'completed'
        assert len(summary['evaluations']) == 78
        expected_jobs = set()
        for seed in range(3):
            for budget in [16, 64, 256]:
                for arm in ['real8', 'real8_generated']:
                    for p in [.95, .99]:
                        expected_jobs.add((seed, budget, 'full', arm, 'mixed', p, True))
                    if budget == 64:
                        for variant in ['clean', 'pipeline']:
                            expected_jobs.add((seed, budget, 'full', arm, variant, .95, True))
                        expected_jobs.add((seed, budget, 'full', arm, 'mixed', .99, False))
            for mode in ['no_time', 'no_context', 'no_attributes', 'no_history']:
                for p in [.95, .99]:
                    expected_jobs.add((seed, 64, mode, 'real8_generated', 'mixed', p, True))
        actual_jobs = {tuple(row[k] for k in ['seed', 'budget', 'mode', 'arm', 'variant', 'precision', 'safe'])
                       for row in summary['evaluations']}
        assert actual_jobs == expected_jobs
        checked[str(summary_path.relative_to(HERE))] = sha256(summary_path)
        for row in summary['evaluations']:
            path = HERE / row['result']; assert sha256(path) == row['result_sha256']
            checked[row['result']] = sha256(path)
            result = json.loads(path.read_text()); folder = path.parent
            assert sha256(HERE / row['model']) == result['model_sha256'] == row['model_meta']['sha256']
            train_folder = HERE / 'data' / domain / f"seed{row['seed']}_budget{row['budget']}_{row['mode']}"
            assert sha256(train_folder / 'fit.npz') == row['fit_cache_sha256']
            assert sha256(train_folder / 'calibration.npz') == row['calibration_cache_sha256']
            if row['model'] not in verified_models:
                valid = np.load(train_folder / 'calibration.npz')
                saved = np.load((HERE / row['model']).with_suffix('.calibration.npz'))
                np.testing.assert_array_equal(valid['y'], saved['y'])
                np.testing.assert_array_equal(valid['operation'], saved['operation'])
                models = joblib.load(HERE / row['model'])
                probabilities = np.zeros(len(valid['y']))
                with threadpool_limits(limits=4):
                    for op, model in enumerate(models):
                        mask = valid['operation'] == op
                        if mask.any(): probabilities[mask] = model.predict_proba(valid['x'][mask])[:, 1]
                np.testing.assert_allclose(probabilities, saved['probability'], rtol=0, atol=1e-12)
                for p in [.95, .99]:
                    thresholds, details = select_thresholds(valid['y'], probabilities, valid['operation'], p)
                    assert thresholds == row['model_meta']['thresholds'][str(p)]['values']
                    assert details == row['model_meta']['thresholds'][str(p)]['details']
                verified_models.add(row['model'])
            for name, digest in result['artifacts'].items(): assert sha256(folder / name) == digest
            key = domain, row['seed'], row['variant']
            if key not in cache:
                source = HERE / 'data' / domain / f"test_seed{row['seed']}_{row['variant']}"
                graph = load_graph(source / 'observed.json.gz')
                reference = json.loads((source / 'reference.json').read_text())
                types = set(reference['config'].get('eligible_types', reference['config'].get('target_types')))
                cache[key] = graph, reference, types
                for name in ['observed.json.gz', 'reference.json']:
                    checked[str((source / name).relative_to(HERE))] = sha256(source / name)
            graph, reference, types = cache[key]
            repaired = load_graph(folder / 'repaired.json.gz')
            clusters = json.loads((folder / 'clusters.json').read_text())
            audit = json.loads((folder / 'audit.json').read_text())
            replay = RepairState(graph)
            replay.apply_batch([Edit(**item['edit']) for item in audit['edits']])
            assert replay.graph == repaired and replay.clusters == clusters
            while replay.log: replay.undo()
            assert replay.graph == graph
            metric = independent_metrics(graph, repaired, reference, types, clusters)
            assert abs(metric['f1'] - result['history']['f1']) < 1e-12
            assert abs(metric['exact'] - result['history']['exact_history_fraction']) < 1e-12
            assert metric['damage'] == result['events']['expanded_new_damage']
            assert metric['recovery'] == result['events']['expanded_repaired_events']
            assert metric['correct_before'] == result['events']['originally_correct_events']
            assert metric['incorrect_before'] == result['events']['originally_incorrect_events']
            assert metric['correct_pairs'] == result['identity']['correct_pairs']
            assert metric['false_pairs'] == result['identity']['predicted_pairs'] - result['identity']['correct_pairs']
            record = {k: row[k] for k in ['seed', 'budget', 'mode', 'arm', 'variant', 'precision', 'safe']}
            record.update({'domain': domain, **metric,
                           'damage_rate': metric['damage'] / max(1, metric['correct_before']),
                           'recovery_rate': metric['recovery'] / max(1, metric['incorrect_before']),
                           'false_pairs': result['identity']['predicted_pairs'] - result['identity']['correct_pairs'],
                           'correct_pairs': result['identity']['correct_pairs'],
                           'train_seconds': row['model_meta']['seconds'],
                           'train_cpu_seconds': row['model_meta']['cpu_seconds'],
                           'inference_seconds': result['seconds'] + result['feature_seconds'],
                           'inference_cpu_seconds': result['cpu_seconds'] + result['feature_cpu_seconds']})
            records.append(record)
        # Same real input bytes and same calibration input bytes for both arms.
        pairs = defaultdict(list)
        for row in summary['evaluations']:
            if row['mode'] == 'full' and row['variant'] == 'mixed' and row['safe'] and row['precision'] == .95:
                pairs[row['seed'], row['budget']].append(row)
        assert len(pairs) == 9
        for rows in pairs.values():
            assert len(rows) == 2
            for field in ['fit_cache_sha256', 'calibration_cache_sha256']:
                assert len({r[field] for r in rows}) == 1
    groups = defaultdict(list)
    keys = ['domain', 'budget', 'mode', 'arm', 'variant', 'precision', 'safe']
    fields = ['f1', 'exact', 'damage', 'recovery', 'damage_rate', 'recovery_rate', 'false_pairs', 'correct_pairs',
              'train_seconds', 'train_cpu_seconds', 'inference_seconds', 'inference_cpu_seconds']
    for row in records: groups[tuple(row[k] for k in keys)].append(row)
    aggregates = []
    for key, rows in groups.items():
        assert len(rows) == 3 and {r['seed'] for r in rows} == {0, 1, 2}
        aggregates.append({**dict(zip(keys, key)), 'n': 3,
                           **{field: {'mean': float(np.mean([r[field] for r in rows])),
                                      'min': float(min(r[field] for r in rows)), 'max': float(max(r[field] for r in rows))} for field in fields}})
    paired = []
    for domain in ['bpi2017', 'bpi2019']:
        for budget in [16, 64, 256]:
            for precision in [.95, .99]:
                rows = [r for r in records if r['domain'] == domain and r['budget'] == budget and r['precision'] == precision
                        and r['mode'] == 'full' and r['variant'] == 'mixed' and r['safe']]
                differences, dominates, real_dominates = [], [], []
                for seed in range(3):
                    real = next(r for r in rows if r['seed'] == seed and r['arm'] == 'real8')
                    mix = next(r for r in rows if r['seed'] == seed and r['arm'] == 'real8_generated')
                    differences.append({f: mix[f] - real[f] for f in ['f1', 'damage', 'recovery', 'damage_rate', 'recovery_rate']})
                    dominates.append(mix['recovery'] >= real['recovery'] and mix['damage'] <= real['damage']
                                     and (mix['recovery'] > real['recovery'] or mix['damage'] < real['damage']))
                    real_dominates.append(real['recovery'] >= mix['recovery'] and real['damage'] <= mix['damage']
                                          and (real['recovery'] > mix['recovery'] or real['damage'] < mix['damage']))
                paired.append({'domain': domain, 'budget': budget, 'precision': precision,
                               'differences': differences, 'generated_dominates_seeds': sum(dominates),
                               'real_dominates_seeds': sum(real_dominates)})
    with (HERE / 'per_trial.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    write(HERE / 'analysis.json', {'status': 'complete_all_156_followup_conditions_audited',
                                  'conditions': len(records), 'models_with_replayed_calibration': len(verified_models),
                                  'aggregates': aggregates, 'paired': paired,
                                  'checks': ['saved-edit replay', 'full undo', 'protected event content',
                                             'independent history metric', 'independent cluster-expanded damage/recovery',
                                             'identical real and calibration caches across arms',
                                             'all model calibration scores and thresholds independently replayed'],
                                  'checked_files': checked, 'analysis_code_sha256': sha256(__file__)})
    print(json.dumps({'conditions': len(records), 'groups': len(aggregates), 'paired': paired}, indent=2))


if __name__ == '__main__':
    main()
