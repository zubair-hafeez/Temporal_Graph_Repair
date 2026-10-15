"""Evaluator-only application metrics. Never import these into predictors."""
from collections import defaultdict
import numpy as np


def traceability(graph, reference, target_types, clusters=None):
    """Retrieve an object's history from its observed identifier.

    Each reference entity is weighted equally, after averaging over its handles.
    Identity reference maps are used only to score the returned event sets.
    """
    clusters = clusters or {o: o for o in graph.objects}
    correct_histories = defaultdict(set)
    handles = defaultdict(list)
    for eid, os in reference['event_participation'].items():
        for o in os:
            if graph.objects[o] in target_types:
                correct_histories[o].add(eid)
    for o, true in reference['identity'].items():
        if graph.objects[o] in target_types:
            handles[true].append(o)
    predicted = defaultdict(set)
    for e in graph.events:
        for o in e.participants:
            if graph.objects[o] in target_types:
                predicted[clusters[o]].add(e.uid)
    per_entity = []
    per_type = defaultdict(list)
    for true, truth_events in correct_histories.items():
        values = []
        for o in handles[true]:
            found = predicted[clusters[o]]
            common = len(found & truth_events)
            precision = common / len(found) if found else 0.0
            recall = common / len(truth_events)
            f1 = 2 * common / (len(found) + len(truth_events))
            values.append((precision, recall, f1, float(found == truth_events)))
        value = np.mean(values, axis=0)
        per_entity.append(value)
        per_type[graph.objects[true]].append(value)
    fields = ['precision', 'recall', 'f1', 'exact_history_fraction']
    result = {k: float(v) for k, v in zip(fields, np.mean(per_entity, axis=0))}
    result['entities'] = len(per_entity)
    result['per_type'] = {typ: {**{k: float(v) for k, v in zip(fields, np.mean(vals, axis=0))},
                               'entities': len(vals)} for typ, vals in per_type.items()}
    return result


def identity_pair_scores(graph, reference, target_types, clusters):
    predicted_groups = defaultdict(list)
    true_groups = defaultdict(list)
    for o, typ in graph.objects.items():
        if typ in target_types:
            predicted_groups[clusters[o]].append(o)
            true_groups[reference['identity'][o]].append(o)
    predicted_count = sum(len(g) * (len(g) - 1) // 2 for g in predicted_groups.values())
    true_count = sum(len(g) * (len(g) - 1) // 2 for g in true_groups.values())
    correct = 0
    for g in predicted_groups.values():
        parts = defaultdict(int)
        for o in g:
            parts[reference['identity'][o]] += 1
        correct += sum(n * (n - 1) // 2 for n in parts.values())
    return {'predicted_pairs': predicted_count, 'true_pairs': true_count, 'correct_pairs': correct,
            'precision': correct / predicted_count if predicted_count else None,
            'recall': correct / true_count if true_count else None,
            'f1': 2 * correct / (predicted_count + true_count) if predicted_count + true_count else None}
