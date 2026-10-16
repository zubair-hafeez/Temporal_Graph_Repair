"""A schema-aware control that commits root replacements atomically.

This is a conservative application control, not a new learning architecture.
It requires an independently declared exactly-one root relation per event.
It does not infer that requirement from evaluator references. Other target
types and alias edits follow the original decoder's rules.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import time

import numpy as np

from tgrepair.action_features import propose_actions
from tgrepair.edits import RepairState


def candidate_transactions(observation, scores, thresholds, root_type, applied, applied_keys):
    """Group a sole-root removal with a supported replacement, or abstain."""
    events = {e.uid: e for e in observation.graph.events}
    eligible = []
    root_actions = defaultdict(lambda: {0: [], 1: []})
    for i in np.flatnonzero(scores >= np.asarray(thresholds)[observation.operations]):
        edit = observation.edits[i]
        op = observation.operations[i]
        if edit in applied:
            continue
        if op != 2:
            inverse = ('REMOVE_PARTICIPANT' if op == 0 else 'ADD_PARTICIPANT', edit.source, edit.target)
            if inverse in applied_keys:
                continue
        if op != 2 and observation.graph.objects[edit.target] == root_type:
            root_actions[edit.source][int(op)].append((edit, float(scores[i])))
        else:
            eligible.append([(edit, float(scores[i]))])
    for uid, actions in root_actions.items():
        current = {o for o in events[uid].participants if observation.graph.objects[o] == root_type}
        adds = sorted(actions[0], key=lambda x: (-x[1], x[0].target))
        removes = sorted(actions[1], key=lambda x: (-x[1], x[0].target))
        if not current:
            if adds:
                eligible.append([adds[0]])
        elif len(current) == 1:
            # Both classifiers must support the change. A confident deletion
            # alone cannot leave an event without its required root relation.
            if adds and removes:
                eligible.append([removes[0], adds[0]])
        elif removes:
            # A pre-existing multiplicity violation can be reduced one step;
            # never create a new one or guess which unsupported ID to retain.
            eligible.append([removes[0]])
    return sorted(eligible, key=lambda transaction: -min(score for _, score in transaction))


def decode_atomic_root(graph, target_types, scorer, thresholds, root_type,
                       rounds=4, participant_budget=128, pair_budget=64, progress=None):
    if root_type not in target_types:
        raise ValueError('Root contract must name a repair target type')
    start = time.monotonic()
    state = RepairState(graph)
    applied, applied_keys, history = set(), set(), []
    for iteration in range(rounds):
        observation = propose_actions(state, target_types, seed=0, event_budget=None,
                                      participant_budget=participant_budget, pair_budget=pair_budget)
        scores = scorer(observation)
        if len(scores) != len(observation.edits) or not np.isfinite(scores).all():
            raise ValueError('Invalid action scores')
        proposals = candidate_transactions(observation, scores, thresholds, root_type, applied, applied_keys)
        selected, used_events, touched, merged = [], set(), set(), set()
        accepted_transactions = 0
        for transaction in proposals:
            edit = transaction[0][0]
            if edit.operation == 'MERGE_ALIASES':
                endpoints = {edit.source, edit.target}
                if endpoints & touched:
                    continue
                touched.update(endpoints)
                merged.update(endpoints)
            else:
                event_key = (edit.source, state.graph.objects[edit.target])
                targets = {action.target for action, _ in transaction}
                if event_key in used_events or targets & merged:
                    continue
                used_events.add(event_key)
                touched.update(targets)
            selected.extend(transaction)
            accepted_transactions += 1
        state.apply_batch([edit for edit, _ in selected])
        for edit, _ in selected:
            applied.add(edit)
            applied_keys.add((edit.operation, edit.source, edit.target))
        history.append({'round': iteration + 1, 'stage': 'atomic_root', 'candidates': len(scores),
                        'transactions': accepted_transactions,
                        'counts': dict(Counter(edit.operation for edit, _ in selected)),
                        'edits': [{'edit': asdict(edit), 'score': score} for edit, score in selected]})
        if progress:
            progress({'round': iteration + 1, 'candidates': len(scores), 'accepted': len(selected),
                      'transactions': accepted_transactions, 'seconds': time.monotonic() - start})
        if not selected:
            break
    return state, {'rounds': history, 'seconds': time.monotonic() - start,
                   'stopped': not history[-1]['edits'] if history else True,
                   'edit_count': len(applied), 'thresholds': list(thresholds),
                   'mode': 'atomic_root', 'root_contract': root_type,
                   'participant_budget': participant_budget, 'pair_budget': pair_budget}
