"""Observation-only follow-up features, executable worlds and auditable repair.

This module deliberately does not alter any previously frozen implementation.
References are consumed only by training_labels and evaluation outside this file.
"""
from collections import defaultdict
from dataclasses import asdict
from types import SimpleNamespace
import random
import time

import numpy as np

from tgrepair.data import Event, Graph, anonymize
from tgrepair.corruption import mixed_faults
from tgrepair.pipeline_faults import pipeline_faults
from tgrepair.edits import Edit, RepairState
from tgrepair.matching import ObservedIndex
from tgrepair.entity_features import EntityIndex
from revision.transactional_decoding import candidate_transactions

MODES = ('full', 'no_time', 'no_context', 'no_attributes', 'no_history')
OPS = ('ADD_PARTICIPANT', 'REMOVE_PARTICIPANT', 'MERGE_ALIASES')


def project(graph, target_types, mode):
    if mode not in MODES:
        raise ValueError(mode)
    events = []
    for e in graph.events:
        ids = e.participants
        if mode == 'no_context':
            ids = tuple(o for o in ids if graph.objects[o] in target_types)
        elif mode == 'no_history':
            ids = ()
        events.append(Event(e.uid, 0. if mode == 'no_time' else e.time, e.kind, ids,
                            {} if mode == 'no_attributes' else dict(e.attributes)))
    events.sort(key=lambda e: (e.time, e.uid))
    return Graph(graph.name, dict(graph.objects), events)


def actions(graph, target_types, mode='full', event_limit=None, seed=0):
    """Remove information before retrieval as well as before feature scoring."""
    view = project(graph, target_types, mode)
    index = ObservedIndex(view)
    selected = list(graph.events)
    if event_limit is not None and len(selected) > event_limit:
        selected = random.Random(seed).sample(selected, event_limit)
    edits, xs, ops = [], [], []
    for e in selected:
        for typ in sorted(target_types):
            candidates = index.candidates(e.uid, typ, 128, allow_current=True)
            # Existing associations identify the operation to evaluate even in
            # the history-free control; they do not enter the evidence index.
            candidates = sorted(set(candidates) | {o for o in e.participants if graph.objects[o] == typ})
            for o, row in zip(candidates, index.features(e.uid, candidates, exclude_query=True)):
                op = int(o in e.participants)
                x = np.zeros(28, dtype=np.float32)
                x[:len(row)] = row
                x[25] = index.type_kind_count[typ][e.kind] / max(1, index.kind_count[e.kind])
                x[26] = sum(graph.objects[p] == typ for p in e.participants)
                edits.append(Edit(OPS[op], e.uid, o)); xs.append(x); ops.append(op)
    entities = EntityIndex(view, target_types)
    for a, b in entities.pairs(64):
        x = np.zeros(28, dtype=np.float32)
        x[:25] = entities.features(a, b)
        edits.append(Edit(OPS[2], a, b)); xs.append(x); ops.append(2)
    return SimpleNamespace(graph=graph, evidence=view, edits=edits,
                           action_features=np.clip(np.asarray(xs, dtype=np.float32).reshape(-1, 28), -10, 10),
                           operations=np.asarray(ops, dtype=np.int64))


def training_labels(observation, reference):
    identity = reference['identity']
    existing = {e.uid: {identity[o] for o in e.participants} for e in observation.graph.events}
    truth = {e: set(ids) for e, ids in reference['event_participation'].items()}
    y = []
    for edit, op in zip(observation.edits, observation.operations):
        if op == 2:
            correct = identity[edit.source] == identity[edit.target]
        elif op == 1:
            correct = identity[edit.target] not in truth[edit.source]
        else:
            correct = identity[edit.target] in truth[edit.source] - existing[edit.source]
        y.append(correct)
    return np.asarray(y, dtype=np.float32)


def corrupt(graph, types, seed, variant):
    if variant == 'pipeline':
        return pipeline_faults(graph, seed, types)
    settings = {
        'mixed': (.10, .10, .25, 'contiguous'),
        'independent': (.15, .15, .30, 'independent'),
        'mild': (.04, .04, .10, 'contiguous'),
        'severe': (.20, .20, .40, 'contiguous'),
        'missing': (.25, 0., .15, 'contiguous'),
        'wrong': (.02, .25, .15, 'independent'),
        'fragment': (.04, .04, .60, 'contiguous'),
        'clean': (0., 0., 0., 'contiguous'),
    }
    return mixed_faults(graph, seed, types, *settings[variant])


def workflow(seed, roots=16):
    """Small state-machine executions, with broad timing and observation variation.

    Root/item relationships are structural; no identifier is copied to a field.
    Cancellations terminate an item, retries precede completion, and resources
    queue work. Arrivals overlap and simultaneous recordings are supported.
    """
    rng = random.Random(seed)
    objects = {f'resource{i}': 'Resource' for i in range(rng.randint(3, 8))}
    resources = list(objects)
    available = dict.fromkeys(resources, 0.)
    events = []
    scale = 10 ** rng.uniform(0, 4)
    quantum = rng.choice([0., scale / 2, scale * 4])
    arrival_rate = rng.choice([.15, .5, 2.]) * scale
    counts = defaultdict(int)
    def emit(t, kind, root, item, resource, amount):
        attrs = {}
        if rng.random() < .8:
            attrs['value'] = round(amount * rng.choice([.5, 1., 1., 2.]) + rng.gauss(0, amount * .08), 1)
        if rng.random() < .7:
            attrs['category'] = rng.choice(['standard', 'priority', 'special'])
        if quantum:
            t = round(t / quantum) * quantum
        events.append(Event(f'e{len(events)}', t, kind, (root, item, resource), attrs))
        counts[kind] += 1
    for i in range(roots):
        root = f'root{i}'; objects[root] = 'Root'
        arrival = i * arrival_rate
        for j in range(rng.randint(1, 3)):
            item = f'item{i}_{j}'; objects[item] = 'Item'
            amount = rng.choice([100., 200., 500., 1000.])
            t = arrival + rng.random() * scale
            resource = rng.choice(resources)
            emit(t, 'create', root, item, resource, amount)
            for stage in ('check', 'execute', 'settle'):
                resource = rng.choice(resources)
                t = max(t, available[resource]) + rng.lognormvariate(0, 1) * scale
                available[resource] = t
                if rng.random() < .10:
                    emit(t, 'cancel', root, item, resource, amount)
                    break
                if rng.random() < .30:
                    emit(t, 'retry_' + stage, root, item, resource, amount)
                    t += rng.lognormvariate(0, 1) * scale
                    available[resource] = t
                emit(t, stage, root, item, resource, amount)
    graph = anonymize(Graph('enriched_workflow', objects, events), seed + 117)
    return graph, dict(counts)


def ambiguous_graph(seed=0, pairs=16):
    """An ambiguity challenge with multiple compatible truths, not a hidden winner."""
    objects, events = {}, []
    # Empty histories and indistinguishable catalogs: no action can identify a
    # unique root. This challenge is intentionally narrow and explicit.
    for i in range(pairs * 2):
        objects[f'o{i}'] = 'Root'
    for i in range(pairs):
        events.append(Event(f'e{i}', float(i), 'record', (), {}))
    return anonymize(Graph('unresolvable_catalog', objects, events), seed)


def score(models, observation):
    result = np.zeros(len(observation.edits))
    for op, model in enumerate(models):
        mask = observation.operations == op
        if mask.any():
            result[mask] = model.predict_proba(observation.action_features[mask])[:, 1]
    return result


def decode(observation, scores, thresholds, root_type, safe=True):
    """One simultaneous proposal pass; original event content is never edited.

    Root substitutions are atomic. Safe mode rejects tied additions and weak,
    non-mutual or co-occurring identity merges. Thresholds are calibrated before
    testing. The guard has no access to evaluator identities or references.
    """
    start = time.perf_counter()
    adjusted = scores.copy()
    rejected = defaultdict(int)
    if safe:
        additions, merges = defaultdict(list), defaultdict(list)
        pair_events = set()
        for e in observation.evidence.events:
            ids = sorted(e.participants)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    pair_events.add((a, b))
        for i, (edit, op) in enumerate(zip(observation.edits, observation.operations)):
            if op == 0:
                additions[edit.source, observation.graph.objects[edit.target]].append(i)
            elif op == 2:
                merges[edit.source].append(i); merges[edit.target].append(i)
        for indices in additions.values():
            if len(indices) > 1:
                ranked = sorted(indices, key=lambda i: -scores[i])
                if scores[ranked[0]] - scores[ranked[1]] <= 1e-8:
                    adjusted[indices] = -1.; rejected['tied_addition'] += len(indices)
        best = {}
        for obj, indices in merges.items():
            ranked = sorted(indices, key=lambda i: -scores[i])
            if len(ranked) == 1 or scores[ranked[0]] - scores[ranked[1]] >= .05:
                best[obj] = ranked[0]
        for i, (edit, op) in enumerate(zip(observation.edits, observation.operations)):
            if op != 2:
                continue
            reason = None
            if tuple(sorted((edit.source, edit.target))) in pair_events:
                reason = 'cooccurring_handles'
            elif scores[i] < .99:
                reason = 'merge_floor'
            elif best.get(edit.source) != i or best.get(edit.target) != i:
                reason = 'non_mutual_or_low_margin'
            if reason:
                adjusted[i] = -1.; rejected[reason] += 1
    proposals = candidate_transactions(observation, adjusted, thresholds, root_type, set(), set())
    selected, used_events, touched, merged = [], set(), set(), set()
    for transaction in proposals:
        edit = transaction[0][0]
        if edit.operation == OPS[2]:
            ends = {edit.source, edit.target}
            if ends & touched:
                continue
            touched.update(ends); merged.update(ends)
        else:
            key = edit.source, observation.graph.objects[edit.target]
            targets = {e.target for e, _ in transaction}
            if key in used_events or targets & merged:
                continue
            used_events.add(key); touched.update(targets)
        selected.extend(transaction)
    state = RepairState(observation.graph)
    state.apply_batch([edit for edit, _ in selected])
    audit = {'mode': 'single_pass_atomic_root', 'safe': safe,
             'thresholds': list(thresholds), 'rejected': dict(rejected),
             'edits': [{'edit': asdict(edit), 'score': float(s)} for edit, s in selected],
             'undo': [{k: v for k, v in record.items() if k != 'edits'} for record in state.log],
             'seconds': time.perf_counter() - start}
    return state, audit
