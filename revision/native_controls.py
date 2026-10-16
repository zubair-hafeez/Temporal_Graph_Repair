"""Shared-input controls for the closed-case missing-ID development track."""
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from revision.here_benchmark import project_single_case, prepare_records
from revision.budget_learning import fit_operation
from tgrepair.data import Graph, load_graph, save_graph, sha256
from tgrepair.matching import ObservedIndex
from tgrepair.metrics import traceability
from tgrepair.action_features import ACTION_DIM

ROOT = Path(__file__).resolve().parents[1]


def action_features(index, event, candidates, root_type):
    raw = index.features(event.uid, candidates, exclude_query=True)
    x = np.zeros((len(candidates), ACTION_DIM), dtype=np.float32)
    x[:, :raw.shape[1]] = raw
    x[:, 25] = index.type_kind_count[root_type][event.kind] / max(1, index.kind_count[event.kind])
    # Known-label supervision simulates an actual missing-ID query. The
    # candidate profile excludes this event and slot occupancy is set to zero.
    x[:, 26] = 0.
    return np.clip(x, -10, 10)


def known_training_examples(observed, root_type, seed, event_limit=4096):
    """Only observed, trusted case IDs produce target-local fitting labels."""
    known = [e for e in observed.events if any(observed.objects[o] == root_type for o in e.participants)]
    if len(known) > event_limit:
        known = random.Random(126000 + seed).sample(known, event_limit)
    index = ObservedIndex(observed)
    xs, ys = [], []
    for e in known:
        case = next(o for o in e.participants if observed.objects[o] == root_type)
        candidates = index.candidates(e.uid, root_type, 128, allow_current=True)
        # Include a labelled positive that retrieval could omit; all supervised
        # controls have this same complete known assignment available.
        if case not in candidates:
            candidates = list(candidates) + [case]
        xs.append(action_features(index, e, candidates, root_type))
        ys.append(np.asarray([o == case for o in candidates], dtype=np.float32))
    return np.concatenate(xs), np.concatenate(ys), {'known_query_events_used': len(known)}


def evaluate(domain, seed=0, missing_rate=.1, max_roots=512):
    start = time.monotonic()
    source = ROOT / 'revision/data/cohorts_v1' / domain / 'development/reference_graph.json.gz'
    manifest = json.loads((ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json').read_text())
    root_type = manifest['root_type']
    clean = project_single_case(load_graph(source), root_type, max_roots)
    rows, hidden, truth, number_case, observed, reference = prepare_records(clean, root_type, missing_rate, seed)
    name = f'native_controls_development_{domain}_seed{seed}_missing{int(missing_rate * 100)}_roots{max_roots}_v1'
    destination = ROOT / 'revision/reports' / (name + '.json')
    if destination.exists():
        raise FileExistsError(destination)
    artifacts = ROOT / 'revision/data/native_controls' / name
    artifacts.mkdir(parents=True)
    save_graph(observed, artifacts / 'observed.json.gz')
    index = ObservedIndex(observed)
    model_names = {'generated_original': 'edit_feature_v2', 'generated_varied': 'edit_feature_attributes_v3'}
    models = {key: joblib.load(ROOT / 'checkpoints' / (model + '.joblib'))[0]
              for key, model in model_names.items()}
    with threadpool_limits(limits=4):
        x, y, counts = known_training_examples(observed, root_type, seed)
        models['known_labels_only'], fit_counts = fit_operation(x, y, seed=seed)
        report = {'status': 'running_native_controls_development', 'name': name,
                  'domain': domain, 'seed': seed, 'missing_rate': missing_rate, 'max_roots': max_roots,
                  'source_sha256': sha256(source), 'implementation_sha256': sha256(__file__),
                  'observed_graph_sha256': sha256(artifacts / 'observed.json.gz'),
                  'generated_model_hashes': {key: sha256(ROOT / 'checkpoints' / (model + '.joblib'))
                                             for key, model in model_names.items()},
                  'observed_events': len(observed.events), 'hidden_events': len(hidden),
                  'known_label_training': {**counts, **fit_counts}, 'results': {},
                  'contract': 'Activity, resource, timestamp and existing case memberships only. No child IDs, vendor context, copied attributes, wrong joins, or aliases. Only missing IDs are modified.',
                  'limits': ['Development comparisons, not fresh confirmation.',
                             'Top-one scoring uses the known single-case task contract; it is not the full iterative repair decoder.',
                             'The target-local boosted control uses up to 4096 known events; HERE uses all remaining known events. Report this unequal fitting sample count.',
                             'Generated-only scorers use no fitting target labels beyond observed memberships.']}
        predictions = {key: {} for key in models}
        predictions['nearest_event'] = {}
        candidate_hits = 0
        # Nearest observed event is a transparent strong temporal reference.
        known_events = [e for e in observed.events if any(observed.objects[o] == root_type for o in e.participants)]
        known_times = np.asarray([e.time for e in known_events])
        for e in observed.events:
            if e.uid not in hidden:
                continue
            candidates = index.candidates(e.uid, root_type, 128, allow_current=True)
            candidate_hits += int(truth[e.uid] in candidates)
            features = action_features(index, e, candidates, root_type)
            for key, model in models.items():
                if candidates:
                    scores = model.predict_proba(features)[:, 1]
                    # Candidate order itself comes from observed retrieval, not
                    # evaluator truth; stable argmax handles score ties.
                    predictions[key][e.uid] = candidates[int(np.argmax(scores))]
            location = int(np.searchsorted(known_times, e.time))
            possible = [j for j in [location - 1, location] if 0 <= j < len(known_events)]
            if possible:
                nearest = min(possible, key=lambda j: (abs(known_times[j] - e.time), known_events[j].uid))
                predictions['nearest_event'][e.uid] = next(o for o in known_events[nearest].participants
                                                          if observed.objects[o] == root_type)
        for key, predicted in predictions.items():
            events = [replace(e, participants=tuple(sorted(set(e.participants) | ({predicted[e.uid]} if e.uid in predicted else set()))))
                      for e in observed.events]
            graph = Graph(observed.name + '_' + key, dict(observed.objects), events)
            correct = sum(predicted.get(uid) == truth[uid] for uid in hidden)
            report['results'][key] = {'correct': correct, 'assigned': len(predicted), 'hidden': len(hidden),
                                       'accuracy_all_missing': correct / max(1, len(hidden)),
                                       'history': traceability(graph, reference, {root_type})}
        report['candidate_recall'] = candidate_hits / max(1, len(hidden))
        report['seconds'] = time.monotonic() - start
        report['status'] = 'completed_native_controls_development'
        (artifacts / 'predictions.json').write_text(json.dumps(predictions) + '\n')
        report['predictions_sha256'] = sha256(artifacts / 'predictions.json')
        destination.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'name': name, 'candidate_recall': report['candidate_recall'],
                          'accuracy': {key: result['accuracy_all_missing'] for key, result in report['results'].items()},
                          'seconds': report['seconds']}), flush=True)

