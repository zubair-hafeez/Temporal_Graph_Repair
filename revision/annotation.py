"""Verified-history budgets with an explicit label access boundary.

Only ``make_verified_view`` may access the full evaluator reference. Feature and
label construction receive a restricted view containing exactly the requested
histories, all their aliases, and their member events. Unverified identities are
not visible to the learner, even when they appear as candidate negatives.
"""
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from tgrepair.action_features import ACTION_DIM
from tgrepair.corruption import mixed_faults
from tgrepair.data import load_graph, save_graph, sha256
from tgrepair.entity_features import EntityIndex
from tgrepair.matching import ObservedIndex

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class VerifiedView:
    entities: set
    alias_owner: dict
    event_memberships: dict
    root_count: int


def make_verified_view(graph, reference, bundles, selected_roots, target_types):
    """Simulate complete verification of selected bundles; disclose its cost."""
    selected = set().union(*(set(bundles[r]) for r in selected_roots))
    owners = {handle: owner for handle, owner in reference['identity'].items() if owner in selected}
    memberships = {}
    for eid, members in reference['event_participation'].items():
        targets = {o for o in members if graph.objects[o] in target_types}
        if targets & selected:
            if not targets <= selected:
                raise ValueError('A verified bundle crosses an unverified target history')
            memberships[eid] = targets
    return VerifiedView(selected, owners, memberships, len(selected_roots))


def build_examples(graph, verified, target_types, event_limit=4096, seed=177):
    """Build labels without full identity maps or unverified event references."""
    index = ObservedIndex(graph)
    events = [e for e in graph.events if e.uid in verified.event_memberships]
    total_events = len(events)
    if len(events) > event_limit:
        events = random.Random(seed).sample(events, event_limit)
    arrays, labels, operations = [], [], []
    for event in events:
        truth = verified.event_memberships[event.uid]
        existing = {verified.alias_owner[o] for o in event.participants if o in verified.alias_owner}
        for typ in sorted(target_types):
            candidates = index.candidates(event.uid, typ, 128, allow_current=True)
            rows = index.features(event.uid, candidates, exclude_query=True)
            for obj, row in zip(candidates, rows):
                op = int(obj in event.participants)
                owner = verified.alias_owner.get(obj)
                label = owner not in truth if op else owner in (truth - existing)
                x = np.zeros(ACTION_DIM, dtype=np.float32)
                x[:len(row)] = row
                x[25] = index.type_kind_count[typ][event.kind] / max(1, index.kind_count[event.kind])
                x[26] = sum(graph.objects[o] == typ for o in event.participants)
                arrays.append(x)
                labels.append(label)
                operations.append(op)
    entities = EntityIndex(graph, target_types)
    anchors = sorted(o for o in entities.profile if o in verified.alias_owner)
    pairs = set(entities.pairs(64, anchors))
    retrieved_count = len(pairs)
    handles = defaultdict(list)
    for handle in anchors:
        handles[verified.alias_owner[handle]].append(handle)
    for group in handles.values():
        pairs.update(tuple(sorted(pair)) for pair in combinations(group, 2))
    for a, b in sorted(pairs):
        if a not in verified.alias_owner and b not in verified.alias_owner:
            raise AssertionError('Unverified identity pair entered supervision')
        x = np.zeros(ACTION_DIM, dtype=np.float32)
        x[:25] = entities.features(a, b)
        owner_a, owner_b = verified.alias_owner.get(a), verified.alias_owner.get(b)
        label = owner_a is not None and owner_b is not None and owner_a == owner_b
        arrays.append(x)
        labels.append(label)
        operations.append(2)
    x = np.clip(np.asarray(arrays, dtype=np.float32).reshape(-1, ACTION_DIM), -10, 10)
    y = np.asarray(labels, dtype=np.float32)
    op = np.asarray(operations, dtype=np.int64)
    counts = {'verified_root_bundles': verified.root_count, 'verified_target_entities': len(verified.entities),
              'verified_alias_handles': len(verified.alias_owner), 'verified_events': total_events,
              'sampled_events': len(events), 'actions': len(y), 'identity_pairs': len(pairs),
              'positive_pairs_added_from_verified_histories': len(pairs) - retrieved_count,
              'positives_by_operation': [int(y[op == i].sum()) for i in range(3)],
              'examples_by_operation': [int((op == i).sum()) for i in range(3)]}
    return x, y, op, counts


def translate_bundles(clean, reference, bundles):
    original_events = {e.uid: e for e in clean.events}
    mapping = {}
    for eid, origin in reference['event_origin'].items():
        before = original_events[origin].participants
        after = reference['event_participation'][eid]
        if len(before) != len(after):
            raise AssertionError('Reference participation order changed')
        for a, b in zip(before, after):
            if a in mapping and mapping[a] != b:
                raise AssertionError('Inconsistent canonical mapping')
            mapping[a] = b
    return {mapping[r]: sorted(mapping[o] for o in members) for r, members in bundles.items()}


def allocate(roots, allocation_seed, budgets=(16, 64, 256)):
    ranked = sorted(roots, key=lambda r: hashlib.sha256(f'revision_budget:{allocation_seed}:{r}'.encode()).hexdigest())
    result = {}
    # Every fourth root belongs to calibration at every nested budget.
    # Increasing the budget never changes an earlier root's role.
    for budget in budgets:
        chosen = ranked[:budget]
        if len(chosen) != budget:
            raise ValueError('Insufficient annotation roots')
        result[budget] = {'fit': set(chosen[i] for i in range(budget) if i % 4 != 3),
                          'calibration': set(chosen[i] for i in range(budget) if i % 4 == 3)}
    return result


def prepare(domain, seed):
    start = time.monotonic()
    source = ROOT / 'revision/data/cohorts_v1' / domain / 'annotation'
    destination = ROOT / 'revision/data/annotation_v1' / domain / f'seed{seed}'
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json').read_text())
    types = set(manifest['target_types'])
    clean = load_graph(source / 'reference_graph.json.gz')
    bundles = json.loads((source / 'bundles.json').read_text())
    graph, reference = mixed_faults(clean, 121000 + seed, types)
    mapped = translate_bundles(clean, reference, bundles)
    allocations = allocate(mapped, 122000 + seed)
    destination.mkdir(parents=True)
    save_graph(graph, destination / 'observed.json.gz')
    # Full simulation references stay in a separately named evaluator file.
    (destination / 'evaluator_reference.json').write_text(json.dumps(reference) + '\n')
    report = {'status': 'budgeted_annotation_cache', 'domain': domain, 'seed': seed,
              'source_graph_sha256': sha256(source / 'reference_graph.json.gz'),
              'cohort_manifest_sha256': sha256(ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json'),
              'implementation_sha256': sha256(__file__), 'annotation_budgets': {},
              'contract': 'Total selected root bundles include fitting and calibration. All other histories are unlabelled context. Costs are simulated access to released references, not measured annotator time.'}
    for budget, split in allocations.items():
        report['annotation_budgets'][str(budget)] = {}
        for role, roots in split.items():
            verified = make_verified_view(graph, reference, mapped, roots, types)
            x, y, op, counts = build_examples(graph, verified, types)
            filename = f'{role}_{budget}.npz'
            np.savez_compressed(destination / filename, x=x, y=y, operation=op)
            counts.update({'file': filename, 'sha256': sha256(destination / filename),
                           'selected_root_ids_sha256': hashlib.sha256('\n'.join(sorted(roots)).encode()).hexdigest()})
            report['annotation_budgets'][str(budget)][role] = counts
            print(json.dumps({'domain': domain, 'seed': seed, 'budget': budget,
                              'role': role, 'counts': counts, 'seconds': time.monotonic() - start}), flush=True)
        (destination / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    report['seconds'] = time.monotonic() - start
    (destination / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')

