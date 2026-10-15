"""Native OCEL observations with a narrow, audited event-field allowlist.

Original identifiers and object/object links never leave this adapter. Source
metadata remains in a separate audit, not in the graph passed to a learner.
"""
from collections import Counter
from datetime import datetime
import gzip
import json
import math
import re

from .data import Event, Graph, anonymize


NATIVE_FIELDS = frozenset({
    'amount', 'quantity', 'currency', 'unit', 'actor_role', 'resource_calendar',
    'location', 'is_exception', 'exception_reason', 'reason', 'changed_field',
})


def adapt_ocel(raw, name, start_fraction=.25, stop_fraction=.5, seed=61001):
    if not 0 <= start_fraction < stop_fraction <= 1:
        raise ValueError('Expected a nonempty fractional time window')
    objects = {o['id']: o['type'] for o in raw['objects']}
    if len(objects) != len(raw['objects']):
        raise ValueError('Duplicate source object identifiers')
    rows = raw['events']
    if not rows or len({e['id'] for e in rows}) != len(rows):
        raise ValueError('Expected nonempty events with unique identifiers')
    def timestamp(event):
        value = datetime.fromisoformat(event['time'].replace('Z', '+00:00'))
        if value.tzinfo is None:
            raise ValueError('Source timestamp must declare a timezone')
        return value.timestamp()
    timed = [(timestamp(e), e) for e in rows]
    ordered_times = sorted(t for t, _ in timed)
    lower = ordered_times[min(len(rows)-1, int(len(rows)*start_fraction))]
    upper = (ordered_times[int(len(rows)*stop_fraction)] if stop_fraction < 1 else None)
    selected = [(t, e) for t, e in timed if t >= lower and (upper is None or t < upper)]
    if not selected:
        raise ValueError('Time window has no events after preserving tied timestamps')
    used = set(); actors = {}; output = []
    kept = Counter(); dropped = Counter(); identifier_values = Counter()
    raw_ids = set(objects)
    for time, event in selected:
        participants = set(r['objectId'] for r in event['relationships'])
        if not participants <= raw_ids:
            raise ValueError('Event references an undeclared object')
        attrs = {}
        seen = set()
        for item in event.get('attributes', []):
            field, value = item['name'], item['value']
            if field in seen:
                raise ValueError('Repeated event attribute name')
            seen.add(field)
            if field == 'actor' and isinstance(value, str) and value:
                # A native actor identifier is context; target IDs cannot become
                # an uncorrupted resource-side copy of their own membership.
                if value in raw_ids:
                    identifier_values[field] += 1
                else:
                    actor = '__native_actor__:' + value
                    if actor in objects:
                        raise ValueError('Actor namespace collision')
                    actors[actor] = 'NativeActor'
                    participants.add(actor)
                continue
            if field not in NATIVE_FIELDS or value is None or not isinstance(value, (str, int, float, bool)):
                dropped[field] += 1
                continue
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Nonfinite native event value')
            if isinstance(value, str) and (value in raw_ids or
                    raw_ids.intersection(re.findall(r'[A-Za-z0-9][A-Za-z0-9_.:-]*', value))):
                identifier_values[field] += 1
                continue
            attrs[field] = value
            kept[field] += 1
        used.update(participants)
        output.append(Event(event['id'], time, event['type'], tuple(sorted(participants)), attrs))
    catalog = {o: objects[o] if o in objects else actors[o] for o in sorted(used)}
    graph = anonymize(Graph(name, catalog, output), seed)
    audit = {
        'source_events': len(rows), 'source_objects': len(objects),
        'window': {'start_fraction': start_fraction, 'stop_fraction': stop_fraction,
                   'start_inclusive_utc_seconds': lower, 'stop_exclusive_utc_seconds': upper,
                   'selection': 'timestamp only; ties remain together; histories may be censored'},
        'native_fields_kept': dict(sorted(kept.items())),
        'fields_dropped': dict(sorted(dropped.items())),
        'identifier_values_dropped': dict(sorted(identifier_values.items())),
        'native_actor_objects': len(actors),
        'excluded_relations': ['object-object links', 'event-object qualifiers'],
        'catalog': 'Only handles referenced in the selected available window, before faults',
        'graph': graph.summary(),
    }
    return graph, audit


def load_external_ocel(path, name, **kwargs):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return adapt_ocel(json.load(stream), name, **kwargs)
