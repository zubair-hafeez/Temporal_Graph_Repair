"""Prospectively assigned, root-disjoint revision cohorts from public logs."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from tgrepair.confirmation_sources import (
    BPI2017_FIELDS, BPI2019_FIELDS, UNKNOWN, _traces, _trace_id,
    _xes_value, _rank, _list_hash, _ocel_array,
)
from tgrepair.data import Event, Graph, anonymize, save_graph, sha256

ROOT = Path(__file__).resolve().parents[1]


def assign_partitions(roots, root_children, excluded_roots, excluded_children, domain, plan):
    owners = defaultdict(set)
    for root, children in root_children.items():
        for child in children:
            owners[child].add(root)
    shared = {child for child, rs in owners.items() if len(rs) > 1}
    crossed = {r for r in roots if root_children.get(r, set()) & (excluded_children | shared)}
    eligible = set(roots) - excluded_roots - crossed
    namespace = plan['ranking_namespace'] + ':' + domain
    ranked = sorted(eligible, key=lambda r: _rank(r, namespace))
    wanted = sum(plan['partition_sizes'].values())
    if len(ranked) < wanted:
        raise ValueError(f'{domain}: {len(ranked)} eligible roots, need {wanted}')
    partitions = {}
    position = 0
    for part, size in plan['partition_sizes'].items():
        partitions[part] = set(ranked[position:position + size])
        position += size
    audit = {'source_roots': len(roots), 'excluded_original_roots': len(excluded_roots),
             'excluded_original_children': len(excluded_children), 'shared_target_children': len(shared),
             'roots_crossing_excluded_or_shared_children': len(crossed), 'eligible_roots': len(eligible),
             'excluded_root_ids_sha256': _list_hash(excluded_roots),
             'partitions': {p: {'root_count': len(rs), 'root_ids_sha256': _list_hash(rs)}
                            for p, rs in partitions.items()}}
    all_root_sets = list(partitions.values())
    all_child_sets = [set().union(*(root_children.get(r, set()) for r in rs)) for rs in all_root_sets]
    for i in range(len(all_root_sets)):
        for j in range(i):
            assert not all_root_sets[i] & all_root_sets[j]
            assert not all_child_sets[i] & all_child_sets[j]
    audit['root_and_child_disjointness_verified'] = True
    return partitions, audit


def bpi2017(path, plan):
    names = []
    children = {}
    audited = set()
    for i, trace in enumerate(_traces(path)):
        name = _trace_id(trace)
        names.append(name)
        if i < 500:
            audited.add(name)
        children[name] = {a.get('value') for e in trace if e.tag.rsplit('}', 1)[-1] == 'event'
                          for a in e if a.get('key') == 'OfferID'}
    if len(names) != len(set(names)):
        raise ValueError('Repeated application root')
    original = set(sorted(set(names) - audited,
                          key=lambda r: _rank(r, 'tgrepair_bpi2017_confirmation_v1'))[:1024])
    excluded = audited | original
    excluded_children = set().union(*(children[r] for r in excluded))
    parts, audit = assign_partitions(set(names), children, excluded, excluded_children, 'bpi2017', plan)
    lookup = {r: p for p, rs in parts.items() for r in rs}
    graphs = {p: Graph('bpi2017_revision_' + p, {}, []) for p in parts}
    bundles = {p: {} for p in parts}
    for trace in _traces(path):
        name = _trace_id(trace)
        if name not in lookup:
            continue
        part = lookup[name]
        graph = graphs[part]
        root = 'Application:' + name
        graph.objects[root] = 'Application'
        bundles[part][root] = {root}
        for event in trace:
            if event.tag.rsplit('}', 1)[-1] != 'event':
                continue
            raw = {a.get('key'): a.get('value') for a in event}
            participants = [root]
            if raw.get('OfferID'):
                child = 'Offer:' + raw['OfferID']
                graph.objects[child] = 'Offer'
                participants.append(child)
                bundles[part][root].add(child)
            resource = raw.get('org:resource', '')
            if resource.upper() not in UNKNOWN:
                resource = 'Resource:' + resource
                graph.objects[resource] = 'Resource'
                participants.append(resource)
            attributes = {}
            for a in event:
                if a.get('key') in BPI2017_FIELDS:
                    value = _xes_value(a)
                    if value is not None:
                        attributes[a.get('key')] = value
            timestamp = datetime.fromisoformat(raw['time:timestamp'].replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                raise ValueError('Timezone-free timestamp')
            kind = raw['concept:name'] + '::' + raw.get('lifecycle:transition', '')
            graph.events.append(Event(str(len(graph.events)), timestamp.timestamp(), kind,
                                      tuple(participants), attributes))
    audit['original_confirmation_root_ids_sha256'] = _list_hash(original)
    return graphs, bundles, audit, {'Application', 'Offer'}, 'Application'


def bpi2019(path, plan):
    objects = {o['id']: o['type'] for o in _ocel_array(path, 'objects')}
    if set(objects.values()) != {'PO', 'POItem', 'Resource', 'Vendor'}:
        raise ValueError('Unexpected BPI2019 schema')
    roots = {o for o, t in objects.items() if t == 'PO'}
    children = defaultdict(set)
    audited_targets = set()
    event_count = 0
    # Schema-only first pass. No model or repair outcome is computed.
    for i, event in enumerate(_ocel_array(path, 'events')):
        participants = {r['objectId'] for r in event['relationships']}
        if i < 2000:
            audited_targets.update(o for o in participants if objects[o] in {'PO', 'POItem'})
        parents = participants & roots
        items = {o for o in participants if objects[o] == 'POItem'}
        for parent in parents:
            children[parent].update(items)
            # Multi-root events share a synthetic structural marker so both roots
            # are excluded rather than placing one source event in two partitions.
            if len(parents) > 1:
                children[parent].add('__multi_root_event__:' + event['id'])
        event_count += 1
    original = set(sorted(roots - audited_targets,
                          key=lambda r: _rank(r, 'tgrepair_bpi2019_confirmation_v1'))[:1024])
    excluded = (roots & audited_targets) | original
    excluded_children = (audited_targets - roots) | set().union(*(children[r] for r in excluded))
    parts, audit = assign_partitions(roots, children, excluded, excluded_children, 'bpi2019', plan)
    lookup = {r: p for p, rs in parts.items() for r in rs}
    graphs = {p: Graph('bpi2019_revision_' + p, {}, []) for p in parts}
    bundles = {p: {} for p in parts}
    for event in _ocel_array(path, 'events'):
        participants = tuple(sorted({r['objectId'] for r in event['relationships']}))
        selected_roots = set(participants) & lookup.keys()
        if not selected_roots:
            continue
        if len(selected_roots) != 1:
            raise AssertionError('Unexpected crossing after structural exclusion')
        root = next(iter(selected_roots))
        part = lookup[root]
        graph = graphs[part]
        graph.objects.update({o: objects[o] for o in participants})
        bundles[part].setdefault(root, set()).update(o for o in participants if objects[o] in {'PO', 'POItem'})
        attributes = {}
        for a in event['attributes']:
            value = a['value']
            if a['name'] in BPI2019_FIELDS and isinstance(value, (str, int, float, bool)):
                if not isinstance(value, float) or math.isfinite(value):
                    attributes[a['name']] = value
        timestamp = datetime.fromisoformat(event['time'].replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            raise ValueError('Timezone-free timestamp')
        graph.events.append(Event(event['id'], timestamp.timestamp(), event['type'], participants, attributes))
    audit['source_events_scanned'] = event_count
    audit['original_confirmation_root_ids_sha256'] = _list_hash(original)
    return graphs, bundles, audit, {'PO', 'POItem'}, 'PO'


def prepare(domain):
    plan_path = ROOT / 'revision/docs/COHORT_PLAN_V1.json'
    plan = json.loads(plan_path.read_text())
    original = json.loads((ROOT / 'docs/CONFIRMATION_PROTOCOL_V1.json').read_text())
    spec = original['sources'][domain]
    source = Path(spec['path'])
    if sha256(source) != spec['sha256']:
        raise ValueError('Public source checksum mismatch')
    destination = ROOT / 'revision/data/cohorts_v1' / domain
    if destination.exists():
        raise FileExistsError(destination)
    graphs, bundles, audit, types, root_type = globals()[domain](source, plan)
    destination.mkdir(parents=True)
    audit.update({'domain': domain, 'plan_sha256': sha256(plan_path),
                  'source_sha256': spec['sha256'], 'adapter_sha256': sha256(__file__),
                  'source_path': str(source), 'target_types': sorted(types), 'root_type': root_type,
                  'status': 'prepared_no_revision_repair_outcomes_opened'})
    for part, graph in graphs.items():
        directory = destination / part
        directory.mkdir()
        graph, mapping, _ = anonymize(graph, plan['anonymization_seeds'][part], return_maps=True)
        if len(bundles[part]) != plan['partition_sizes'][part]:
            raise ValueError('Selected root has no retained history; explicit amendment required')
        mapped_bundles = {mapping[r]: sorted(mapping[o] for o in members)
                          for r, members in bundles[part].items()}
        save_graph(graph, directory / 'reference_graph.json.gz')
        (directory / 'bundles.json').write_text(json.dumps(mapped_bundles, sort_keys=True) + '\n')
        details = audit['partitions'][part]
        details.update({'summary': graph.summary(),
                        'graph_sha256': sha256(directory / 'reference_graph.json.gz'),
                        'bundles_sha256': sha256(directory / 'bundles.json'),
                        'outcomes_opened': False})
    report = ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json'
    report.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'domain': domain, 'partitions': {p: {'roots': a['root_count'],
                      'events': a['summary']['events']} for p, a in audit['partitions'].items()},
                      'report': str(report)}), flush=True)

