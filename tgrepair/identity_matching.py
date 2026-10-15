"""Observed temporal/context identity matching, with optional global assignment.

This baseline assumes fragments of a single object occupy nonoverlapping
intervals. It cannot be used as proof that all real aliases obey that rule.
"""
from collections import Counter, defaultdict
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def alias_candidates(graph, target_types, budget=32, use_attributes=False):
    history = defaultdict(list)
    context = defaultdict(Counter)
    context_objects = defaultdict(set)
    profiles = defaultdict(lambda: defaultdict(Counter))
    for e in graph.events:
        for o in e.participants:
            if graph.objects[o] in target_types:
                history[o].append(e.time)
                for key, value in e.attributes.items():
                    profiles[o][key][value] += 1
                for p in e.participants:
                    if graph.objects[p] != graph.objects[o]:
                        context[o][p] += 1
                        context_objects[p].add(o)
    dominant = {o: {key: counts.most_common(1)[0][0] for key, counts in values.items()}
                for o, values in profiles.items()}
    attribute_objects = defaultdict(set)
    for o, values in dominant.items():
        for key, value in values.items():
            attribute_objects[graph.objects[o], key, value].add(o)
    if use_attributes:
        # Robust temporal bounds use records agreeing with the modal attribute
        # profile; this supplies evidence only and does not delete source events.
        robust = defaultdict(list)
        for e in graph.events:
            for o in e.participants:
                if o in dominant and all(e.attributes.get(k) == v for k, v in dominant[o].items()):
                    robust[o].append(e.time)
        for o, times in robust.items():
            history[o] = times
    edge_scores = []
    for o in sorted(history):
        end = history[o][-1]
        candidates = set()
        for p in context[o]:
            candidates.update(context_objects[p])
        if use_attributes:
            for key, value in dominant.get(o, {}).items():
                candidates.update(attribute_objects[graph.objects[o], key, value])
        possible = []
        for other in candidates:
            if graph.objects[o] != graph.objects[other] or history[other][0] <= end:
                continue
            keys = set(context[o]) | set(context[other])
            numerator = sum(min(context[o][p], context[other][p]) / max(1, len(context_objects[p])) for p in keys)
            denominator = sum(max(context[o][p], context[other][p]) / max(1, len(context_objects[p])) for p in keys)
            similarity = numerator / denominator if denominator else 0
            duration = max(1.0, history[o][-1] - history[o][0], history[other][-1] - history[other][0])
            gap = history[other][0] - end
            temporal = 1.0 / (1.0 + gap / duration)
            score = math.sqrt(similarity * temporal)
            if use_attributes:
                common = set(dominant.get(o, {})) & set(dominant.get(other, {}))
                if common:
                    agreement = sum(dominant[o][k] == dominant[other][k] for k in common) / len(common)
                    score = agreement * (0.5 + 0.25 * similarity + 0.25 * temporal)
            possible.append((score, other))
        for score, other in sorted(possible, reverse=True)[:budget]:
            edge_scores.append((o, other, score))
    return edge_scores


def attribute_variant_clusters(graph, target_types):
    """Adaptation of OCIP's exact observed attribute-set grouping.

    Grouping is used here as a repair baseline; the source code reports clone
    detection. No claim of an unchanged reproduction is made.
    """
    profiles = defaultdict(lambda: defaultdict(set))
    for e in graph.events:
        for o in e.participants:
            if graph.objects[o] in target_types:
                for key, value in e.attributes.items():
                    profiles[o][key].add(value)
    representatives = {}
    result = {o: o for o in graph.objects}
    for o in sorted(profiles):
        signature = (graph.objects[o], tuple((k, tuple(sorted(v, key=repr))) for k, v in sorted(profiles[o].items())))
        result[o] = representatives.setdefault(signature, o)
    return result


def match_aliases(graph, edges, target_types, threshold=0.75, global_assignment=True):
    ids = sorted(o for o, typ in graph.objects.items() if typ in target_types)
    lookup = {o: i for i, o in enumerate(ids)}
    eligible = [(a, b, s) for a, b, s in edges if s > threshold]
    if global_assignment:
        # Private dummy columns allow every source fragment to remain unmerged.
        n = len(ids)
        rows = list(range(n))
        cols = list(range(n, 2 * n))
        values = [1.0] * n
        for a, b, score in eligible:
            rows.append(lookup[a]); cols.append(lookup[b]); values.append(1.0 - (score - threshold))
        matrix = coo_matrix((np.asarray(values), (rows, cols)), shape=(n, 2 * n)).tocsr()
        r, c = min_weight_full_bipartite_matching(matrix)
        chosen = [(ids[i], ids[j]) for i, j in zip(r, c) if j < n]
    else:
        used_from, used_to, chosen = set(), set(), []
        for a, b, _ in sorted(eligible, key=lambda x: (-x[2], x[0], x[1])):
            if a not in used_from and b not in used_to:
                chosen.append((a, b)); used_from.add(a); used_to.add(b)
    parent = {o: o for o in graph.objects}

    def root(o):
        while parent[o] != o:
            parent[o] = parent[parent[o]]
            o = parent[o]
        return o

    for a, b in chosen:
        parent[root(b)] = root(a)
    return {o: root(o) for o in graph.objects}
