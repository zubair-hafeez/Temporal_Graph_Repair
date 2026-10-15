"""Strong observation-only candidate retrieval and repair scoring.

Every index is rebuilt after corruption. No constructor takes reference labels.
Feature learning is a matching baseline, not the proposed generative model.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import math

import numpy as np


FEATURE_NAMES = [
    'log_degree', 'log_cooccurrence', 'normalized_cooccurrence', 'max_jaccard',
    'partner_coverage', 'object_time', 'context_time', 'same_activity_time',
    'bracketed', 'activity_likelihood_ratio', 'previous_transition_ratio',
    'next_transition_ratio', 'observed_context_count',
    'attribute_agreement', 'attribute_conflict', 'attribute_coverage',
    'attribute_exact_pattern', 'attribute_numerical_proximity',
]


class ObservedIndex:
    def __init__(self, graph):
        self.graph = graph
        self.events = {e.uid: e for e in graph.events}
        self.hist = defaultdict(list)
        self.hist_ids = defaultdict(list)
        self.pairs = defaultdict(lambda: defaultdict(list))
        self.pair_ids = defaultdict(lambda: defaultdict(list))
        self.activities = defaultdict(Counter)
        self.type_occurrences = defaultdict(list)
        self.activity_occurrences = defaultdict(list)
        self.kind_count = Counter()
        self.type_kind_count = defaultdict(Counter)
        self.type_count = Counter()
        self.transitions = Counter()
        self.transition_from = Counter()
        self.transition_to = Counter()
        self.degree = Counter()
        self.attribute_objects = defaultdict(set)
        self.attribute_samples = defaultdict(list)
        self.by_type = defaultdict(list)
        for o, typ in graph.objects.items():
            self.by_type[typ].append(o)
        for e in graph.events:
            self.kind_count[e.kind] += 1
            for o in e.participants:
                self.hist[o].append((e.time, e.kind))
                self.hist_ids[o].append(e.uid)
                self.degree[o] += 1
                self.activities[o][e.kind] += 1
                typ = graph.objects[o]
                for key, value in e.attributes.items():
                    self.attribute_objects[typ, key, value].add(o)
                self.type_kind_count[typ][e.kind] += 1
                self.type_count[typ] += 1
                self.type_occurrences[typ].append((e.time, o))
                self.activity_occurrences[(typ, e.kind)].append((e.time, o))
                for p in e.participants:
                    if o != p:
                        self.pairs[p][o].append(e.time)
                        self.pair_ids[p][o].append(e.uid)
            for key, value in e.attributes.items():
                if isinstance(value, (int, float)):
                    self.attribute_samples[key].append(value)
        self.attribute_scales = {key: max(1.0, float(np.std(values)))
                                 for key, values in self.attribute_samples.items()}
        durations = []
        for o, hist in self.hist.items():
            typ = graph.objects[o]
            groups = []
            for t, kind in hist:
                if groups and groups[-1][0] == t:
                    groups[-1][1].append(kind)
                else:
                    groups.append((t, [kind]))
            for (t0, previous), (t1, following) in zip(groups, groups[1:]):
                durations.append(t1 - t0)
                weight = 1.0 / (len(previous) * len(following))
                for a in previous:
                    for b in following:
                        self.transitions[typ, a, b] += weight
                        self.transition_from[typ, a] += weight
                        self.transition_to[typ, b] += weight
        self.scale = max(1.0, float(np.median(durations))) if durations else 1.0
        self.times = {o: [t for t, _ in hist] for o, hist in self.hist.items()}
        self.type_times = {k: [t for t, _ in v] for k, v in self.type_occurrences.items()}
        self.activity_times = {k: [t for t, _ in v] for k, v in self.activity_occurrences.items()}
        self.partner_occurrences = {}
        self.partner_times = {}
        for p, pairs in self.pairs.items():
            by_type = defaultdict(list)
            for o, times in pairs.items():
                by_type[graph.objects[o]].extend((t, o) for t in times)
            for typ, rows in by_type.items():
                rows.sort()
                self.partner_occurrences[p, typ] = rows
                self.partner_times[p, typ] = [t for t, _ in rows]
        self.priors = {typ: sorted(os, key=lambda o: (-self.degree[o], o))
                       for typ, os in self.by_type.items()}

    @staticmethod
    def nearest(rows, times, time, limit):
        if not rows:
            return []
        i = bisect_left(times, time)
        left, right = i - 1, i
        seen = set()
        result = []
        inspected = 0
        while len(result) < limit and inspected < max(512, limit * 32) and (left >= 0 or right < len(rows)):
            if right >= len(rows) or (left >= 0 and time - times[left] <= times[right] - time):
                o = rows[left][1]
                left -= 1
            else:
                o = rows[right][1]
                right += 1
            inspected += 1
            if o not in seen:
                result.append(o)
                seen.add(o)
        return result

    def candidates(self, event_uid, object_type, budget=128, allow_current=False):
        e = self.events[event_uid]
        same_type = self.by_type.get(object_type, [])
        current = [o for o in e.participants if self.graph.objects[o] == object_type]
        forbidden = set(e.participants) if not allow_current else set(e.participants) - set(current)
        if len(same_type) <= budget:
            return sorted(o for o in same_type if o not in forbidden)
        cooc = Counter()
        partner_near = []
        partners = [p for p in e.participants if not allow_current or self.graph.objects[p] != object_type]
        for p in partners:
            for o, times in self.pairs.get(p, {}).items():
                if self.graph.objects[o] == object_type and o not in forbidden:
                    cooc[o] += len(times) / math.sqrt(max(1, self.degree[p] * self.degree[o]))
            key = p, object_type
            partner_near.append(self.nearest(self.partner_occurrences.get(key, []),
                                            self.partner_times.get(key, []), e.time, budget))
        streams = [self.nearest(self.type_occurrences.get(object_type, []),
                                self.type_times.get(object_type, []), e.time, budget),
                   self.nearest(self.activity_occurrences.get((object_type, e.kind), []),
                                self.activity_times.get((object_type, e.kind), []), e.time, budget)]
        streams.extend(partner_near)
        for key, value in e.attributes.items():
            streams.insert(0, sorted(self.attribute_objects.get((object_type, key, value), set()),
                                     key=lambda o: (-self.degree[o], o)))
        if allow_current:
            streams.insert(0, current)
        streams.append(sorted(cooc, key=lambda o: (-cooc[o], o)))
        streams.append(self.priors.get(object_type, []))
        selected = set()
        # Round robin prevents one very high-degree participant from taking the pool.
        for rank in range(budget):
            for stream in streams:
                if rank < len(stream) and stream[rank] not in forbidden:
                    selected.add(stream[rank])
                    if len(selected) == budget:
                        return sorted(selected)
        return sorted(selected)

    def proximity(self, times, time):
        if not times:
            return 0.0
        i = bisect_left(times, time)
        gap = min(abs(times[j] - time) for j in (i - 1, i) if 0 <= j < len(times))
        return 1.0 / (1.0 + gap / self.scale)

    def features(self, event_uid, candidates, exclude_query=False):
        e = self.events[event_uid]
        rows = []
        for o in candidates:
            typ = self.graph.objects[o]
            kinds = self.type_kind_count[typ]
            n_events = self.type_count[typ]
            base_kind = (kinds[e.kind] + 1) / (n_events + max(1, len(kinds)))
            deg = self.degree[o]
            hist = self.hist.get(o, [])
            times = self.times.get(o, [])
            partners_visible = list(e.participants)
            contains_query = e.uid in self.hist_ids.get(o, []) if exclude_query else False
            if exclude_query:
                partners_visible = [p for p in e.participants if self.graph.objects[p] != typ]
            if contains_query:
                hist = [h for h, uid in zip(hist, self.hist_ids[o]) if uid != e.uid]
                times = [t for t, _ in hist]
                deg = len(hist)
            insertion = bisect_left(times, e.time)
            cooc, normalized, jaccard, partners, ctx_time = 0, 0.0, 0.0, 0, 0.0
            for p in partners_visible:
                pts = self.pairs.get(p, {}).get(o, [])
                if contains_query:
                    pts = [t for t, uid in zip(pts, self.pair_ids[p][o]) if uid != e.uid]
                n = len(pts)
                cooc += n
                partner_degree = self.degree[p] - int(exclude_query and p in e.participants)
                normalized += n / math.sqrt(max(1, deg * partner_degree))
                jaccard = max(jaccard, n / max(1, deg + partner_degree - n))
                partners += bool(n)
                ctx_time = max(ctx_time, self.proximity(pts, e.time))
            same_activity = [t for t, k in hist if k == e.kind]
            activity_count = self.activities[o][e.kind] - int(contains_query)
            activity_prob = (activity_count + base_kind) / (deg + 1)
            prev_ratio, next_ratio = 0.0, 0.0
            if insertion > 0:
                start = bisect_left(times, times[insertion - 1])
                probs = [(self.transitions[typ, a, e.kind] + base_kind) / (self.transition_from[typ, a] + 1)
                         for _, a in hist[start:insertion]]
                prob = sum(probs) / len(probs)
                prev_ratio = math.log(max(prob / base_kind, 1e-8))
            # Equal-time records are not treated as a strict successor.
            j = bisect_right(times, e.time)
            if j < len(hist):
                end = bisect_right(times, times[j])
                ratios = []
                for _, b in hist[j:end]:
                    base_b = (kinds[b] + 1) / (n_events + max(1, len(kinds)))
                    prob = (self.transitions[typ, e.kind, b] + base_b) / (self.transition_from[typ, e.kind] + 1)
                    ratios.append(prob / base_b)
                next_ratio = math.log(max(sum(ratios) / len(ratios), 1e-8))
            attributes = ([self.events[uid].attributes for uid in self.hist_ids.get(o, [])
                           if not exclude_query or uid != e.uid] if e.attributes else [])
            agreement, conflict, coverage, exact, numerical = 0., 0., 0., 0., 0.
            if e.attributes:
                numeric_count = 0
                for key, value in e.attributes.items():
                    vals = [a[key] for a in attributes if key in a]
                    if vals:
                        match = sum(v == value for v in vals) / len(vals)
                        agreement += match; conflict += 1 - match; coverage += 1
                        if isinstance(value, (int, float)) and all(isinstance(v, (int, float)) for v in vals):
                            numerical += 1 / (1 + abs(value - float(np.median(vals))) / self.attribute_scales[key])
                            numeric_count += 1
                agreement /= len(e.attributes); conflict /= len(e.attributes); coverage /= len(e.attributes)
                exact = sum(all(a.get(k) == v for k, v in e.attributes.items()) for a in attributes) / max(1, len(attributes))
                numerical /= max(1, numeric_count)
            rows.append([math.log1p(deg), math.log1p(cooc), normalized, jaccard,
                         partners / max(1, len(partners_visible)), self.proximity(times, e.time),
                         ctx_time, self.proximity(same_activity, e.time),
                         float(bool(times) and times[0] < e.time < times[-1]),
                         math.log(max(activity_prob / base_kind, 1e-8)), prev_ratio, next_ratio,
                         len(partners_visible), agreement, conflict, coverage, exact, numerical])
        return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))


def fixed_scores(features):
    f = features
    scores = {
        'degree': f[:, 0],
        'cooccurrence': f[:, 1],
        'normalized_cooccurrence': f[:, 2],
        'jaccard': f[:, 3],
        'temporal_nearest': f[:, 5],
        'context_temporal': f[:, 6],
        'activity_temporal': f[:, 7],
        'structural_temporal': f[:, 6] + f[:, 2] + 0.25 * f[:, 5],
        'activity_transition': f[:, 5] + 0.05 * (f[:, 9] + f[:, 10] + f[:, 11]),
    }
    if f.shape[1] >= 18:
        scores['attribute_exact'] = f[:, 16]
        scores['attribute_temporal'] = 2 * f[:, 13] + f[:, 6] + 0.25 * f[:, 5]
    return scores


def expected_topk_recall(scores, positive, k):
    """Fractional credit for score ties; absent truths count as failures."""
    if not len(scores) or k == 0:
        return 0.0
    k_pool = min(k, len(scores))
    cutoff = np.sort(scores)[-k_pool]
    equal = np.isclose(scores, cutoff, atol=1e-8, rtol=1e-7)
    greater = (scores > cutoff) & ~equal
    slots = k_pool - int(greater.sum())
    return float(positive[greater].sum() + slots * positive[equal].sum() / max(1, equal.sum())) / k
