"""Observation faults; reference labels must not be passed to predictors."""
from collections import defaultdict
from dataclasses import dataclass
import random

from .data import Event, Graph, anonymize


@dataclass(frozen=True)
class Hole:
    event_uid: str
    object_type: str
    reference_object: str


def remove_participation(graph, seed, mode, eligible_types, rate=0.2):
    rng = random.Random(seed)
    memberships = defaultdict(list)
    for e in graph.events:
        for o in e.participants:
            if graph.objects[o] in eligible_types:
                memberships[o].append(e.uid)
    hidden = set()
    if mode == 'independent':
        for o in sorted(memberships):
            for eid in memberships[o]:
                if rng.random() < rate:
                    hidden.add((eid, o))
    elif mode == 'contiguous':
        # Remove approximately the same fraction of each object's history.
        # Stochastic rounding includes short-lived objects without forcing a hole.
        for o in sorted(memberships):
            eids = memberships[o]
            expected = rate * len(eids)
            width = int(expected) + (rng.random() < expected % 1)
            if width:
                start = rng.randrange(len(eids) - width + 1)
                hidden.update((eid, o) for eid in eids[start:start + width])
    else:
        raise ValueError(mode)
    observed = Graph(graph.name + f'_{mode}_{seed}', dict(graph.objects),
                     [Event(e.uid, e.time, e.kind, tuple(o for o in e.participants
                                                       if (e.uid, o) not in hidden), dict(e.attributes))
                      for e in graph.events])
    labels = [Hole(eid, graph.objects[o], o) for eid, o in sorted(hidden)]
    observed.validate()
    return observed, labels


def mixed_faults(graph, seed, eligible_types, missing_rate=0.1, wrong_rate=0.1,
                 fragment_rate=0.25, gap_mode='contiguous'):
    """Namespace fragmentation followed by missing and incorrect joins.

    Returns a public graph and a private evaluator dictionary. All handles are
    reshuffled after fault application so new aliases have no naming clue.
    """
    rng = random.Random(seed + 18973)
    occurrences = defaultdict(list)
    for e in graph.events:
        for o in e.participants:
            if graph.objects[o] in eligible_types:
                occurrences[o].append(e.uid)
    objects = dict(graph.objects)
    identity = {o: o for o in objects}
    remap = {}
    fragmentations = []
    for o in sorted(occurrences):
        eids = occurrences[o]
        if len(eids) >= 4 and rng.random() < fragment_rate:
            cut = rng.randrange(1, len(eids))
            alias = 'temporary_alias_' + o
            objects[alias] = objects[o]
            identity[alias] = o
            for eid in eids[cut:]:
                remap[eid, o] = alias
            fragmentations.append((o, alias))
    fragmented = Graph(graph.name, objects,
                       [Event(e.uid, e.time, e.kind,
                              tuple(remap.get((e.uid, o), o) for o in e.participants), dict(e.attributes))
                        for e in graph.events])
    missing, holes = remove_participation(fragmented, seed + 478, gap_mode, eligible_types, missing_rate)
    by_type = defaultdict(list)
    for o, typ in objects.items():
        by_type[typ].append(o)
    wrong = []
    modified = []
    for e in missing.events:
        participants = list(e.participants)
        for i, o in enumerate(participants):
            typ = objects[o]
            if typ in eligible_types and rng.random() < wrong_rate:
                # Equal-type alternatives only; do not accidentally choose an alias
                # of the same true entity or an already present participant.
                options = [p for p in by_type[typ] if identity[p] != identity[o] and p not in participants]
                if options:
                    replacement = rng.choice(options)
                    participants[i] = replacement
                    wrong.append((e.uid, o, replacement))
        modified.append(Event(e.uid, e.time, e.kind, tuple(participants), dict(e.attributes)))
    observed, object_map, event_map = anonymize(Graph(graph.name + '_mixed', objects, modified),
                                               seed + 79571, return_maps=True)
    truth = {
        'event_participation': {event_map[e.uid]: [object_map[o] for o in e.participants] for e in graph.events},
        'identity': {object_map[o]: object_map[identity[o]] for o in objects},
        'event_origin': {event_map[e.uid]: e.uid for e in graph.events},
        'missing': [(event_map[h.event_uid], object_map[h.reference_object]) for h in holes],
        'wrong': [(event_map[eid], object_map[a], object_map[b]) for eid, a, b in wrong],
        'fragmentations': [(object_map[a], object_map[b]) for a, b in fragmentations],
        'config': {'seed': seed, 'missing_rate': missing_rate, 'wrong_rate': wrong_rate,
                   'fragment_rate': fragment_rate, 'gap_mode': gap_mode,
                   'eligible_types': sorted(eligible_types)},
    }
    return observed, truth
