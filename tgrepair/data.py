"""Canonical observations and read-only source adapters.

Source memberships and grouping keys are never retained as event attributes.
The graph supplied to a predictor contains only explicitly observed facts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import io
import json
import random
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Event:
    uid: str
    time: float
    kind: str
    participants: tuple[str, ...]
    attributes: dict = field(default_factory=dict)


@dataclass
class Graph:
    name: str
    objects: dict[str, str]
    events: list[Event]

    def validate(self):
        assert len({e.uid for e in self.events}) == len(self.events)
        assert all(self.events[i].time <= self.events[i + 1].time
                   for i in range(len(self.events) - 1))
        for e in self.events:
            assert len(set(e.participants)) == len(e.participants), e.uid
            assert all(o in self.objects for o in e.participants), e.uid

    def summary(self):
        return {"name": self.name, "events": len(self.events),
                "objects": len(self.objects),
                "object_types": dict(Counter(self.objects.values())),
                "event_types": dict(Counter(e.kind for e in self.events)),
                "participation_links": sum(len(e.participants) for e in self.events)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def anonymize(graph: Graph, seed=31337, return_maps=False):
    """Shuffle ID assignment and time ties independently of source ordering."""
    rng = random.Random(seed)
    keys = sorted(graph.objects)
    handles = list(range(len(keys)))
    rng.shuffle(handles)
    mapping = {k: f'o{v:09d}' for k, v in zip(keys, handles)}
    event_handles = list(range(len(graph.events)))
    rng.shuffle(event_handles)
    events = [Event(f'e{event_handles[i]:09d}', e.time, e.kind,
                    tuple(sorted(mapping[o] for o in e.participants)), dict(e.attributes))
              for i, e in enumerate(graph.events)]
    events.sort(key=lambda e: (e.time, e.uid))
    result = Graph(graph.name, {mapping[o]: t for o, t in graph.objects.items()}, events)
    result.validate()
    if return_maps:
        return result, mapping, {e.uid: f'e{event_handles[i]:09d}' for i, e in enumerate(graph.events)}
    return result


def save_graph(graph, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as z:
        with io.TextIOWrapper(z, encoding='utf-8') as f:
            json.dump(asdict(graph), f, separators=(',', ':'))


def load_graph(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        raw = json.load(f)
    g = Graph(raw['name'], raw['objects'],
              [Event(e['uid'], e['time'], e['kind'], tuple(e['participants']), e.get('attributes', {}))
               for e in raw['events']])
    g.validate()
    return g


def load_logistics(path):
    raw = json.loads(Path(path).read_text())
    objects = {o['id']: o['type'] for o in raw['objects']}
    events = [Event(e['id'], datetime.fromisoformat(e['time'].replace('Z', '+00:00')).timestamp(),
                    e['type'], tuple(sorted(set(r['objectId'] for r in e['relationships']))))
              for e in raw['events']]
    return anonymize(Graph('logistics_development', objects, events))


def partition_case(case_id):
    n = int(hashlib.sha256(('tgrepair_bpi2012_v1:' + case_id).encode()).hexdigest()[:8], 16) % 100
    return 'fit' if n < 70 else 'tune' if n < 85 else 'check'


def load_bpi2012(path, include_attributes=False):
    groups = {k: Graph('bpi2012_development_' + k, {}, []) for k in ('fit', 'tune', 'check')}
    source_event_index = 0
    with gzip.open(path, 'rb') as f:
        iterator = ET.iterparse(f, events=('start', 'end'))
        _, root = next(iterator)
        for tag, trace in iterator:
            if tag != 'end' or trace.tag.rsplit('}', 1)[-1] != 'trace':
                continue
            attributes = {a.get('key'): a.get('value') for a in trace
                          if a.tag.rsplit('}', 1)[-1] != 'event'}
            case_id = attributes['concept:name']
            graph = groups[partition_case(case_id)]
            case = 'case:' + case_id
            graph.objects[case] = 'Application'
            for e in trace:
                if e.tag.rsplit('}', 1)[-1] != 'event':
                    continue
                a = {x.get('key'): x.get('value') for x in e}
                participants = [case]
                resource = a.get('org:resource')
                if resource and resource.upper() not in {'UNKNOWN', 'NONE', 'NULL', 'NAN'}:
                    resource = 'resource:' + resource
                    graph.objects[resource] = 'Resource'
                    participants.append(resource)
                kind = a['concept:name'] + '::' + a.get('lifecycle:transition', '')
                time = datetime.fromisoformat(a['time:timestamp'].replace('Z', '+00:00')).timestamp()
                visible_attributes = {}
                if include_attributes and attributes.get('AMOUNT_REQ') is not None:
                    visible_attributes['RequestedAmount'] = float(attributes['AMOUNT_REQ'])
                graph.events.append(Event(str(source_event_index), time, kind, tuple(participants), visible_attributes))
                source_event_index += 1
            root.clear()
    return {k: anonymize(g) for k, g in groups.items()}
