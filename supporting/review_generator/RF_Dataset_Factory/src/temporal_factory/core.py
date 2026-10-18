"""Common event records, capability dispatch, and scientific task artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from rfdf.storage import atomic_json, assign_split, checksum, safe_id


SCHEMA_VERSION = 'temporal-events/1'


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_execution(execution):
    if execution.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported temporal record version')
    # This validates JSON-compatible finite values, including nested attributes.
    canonical_bytes(execution)
    entities = execution['entities']
    for uid, entity in entities.items():
        if not isinstance(uid, str) or not uid or not entity.get('kind'):
            raise ValueError('Entity identifiers and kinds must be nonempty strings')
        if not math.isfinite(entity['created_at']):
            raise ValueError('Invalid entity creation time')
    seen = set()
    last_time = -math.inf
    for event in execution['events']:
        if event['id'] in seen:
            raise ValueError('Duplicate event identifier')
        seen.add(event['id'])
        if event['time'] < last_time or not math.isfinite(event['time']):
            raise ValueError('Events must be in nondecreasing occurrence time')
        last_time = event['time']
        if event['available_at'] < event['time'] or not math.isfinite(event['available_at']):
            raise ValueError('Recorded evidence cannot arrive before this event occurs')
        participants = event['participants']
        if len(set(participants)) != len(participants):
            raise ValueError('Duplicate participant')
        for uid in participants:
            if uid not in entities or entities[uid]['created_at'] > event['time']:
                raise ValueError('Unknown or not-yet-created participant')
    return execution


class Registry:
    """Explicit module registration; no implicit fallback for missing capabilities."""
    def __init__(self):
        self.worlds = {}
        self.tasks = {}

    def register_world(self, name, factory, capabilities):
        if name in self.worlds:
            raise ValueError('Duplicate world module: ' + name)
        self.worlds[name] = (factory, frozenset(capabilities))

    def register_task(self, name, builder, requires):
        if name in self.tasks:
            raise ValueError('Duplicate task module: ' + name)
        self.tasks[name] = (builder, frozenset(requires))

    def create_world(self, name, recipe):
        return self.worlds[name][0](recipe)

    def build_task(self, name, execution, config):
        validate_execution(execution)
        builder, requires = self.tasks[name]
        available = set(execution['capabilities'])
        if not requires <= available:
            raise ValueError(f'Task {name} requires unsupported capabilities: {sorted(requires - available)}')
        artifact = builder(execution, config)
        if set(artifact) != {'task', 'learner', 'reference', 'metadata'}:
            raise ValueError('Task builder must separate learner, reference, and metadata artifacts')
        forbidden = {'reference', 'faults', 'clean_graph', 'hidden_state', 'seed', 'recipe', 'truth'}
        if forbidden & set(artifact['learner']):
            raise ValueError('Evaluator-only fields found in learner artifact')
        canonical_bytes(artifact)
        artifact['metadata']['execution_sha256'] = fingerprint(execution)
        return artifact


def export_task(root, artifact, group_id, split_seed=0, split=None):
    """Commit separate artifacts and then a hash manifest using factory storage."""
    root = Path(root)
    safe_id(group_id)
    if (root / 'manifest.json').exists():
        raise FileExistsError('A committed task artifact already exists')
    split = split or assign_split(group_id, split_seed, {'train': .7, 'validation': .15, 'test': .15})
    if split not in {'train', 'validation', 'test'}:
        raise ValueError('Invalid split')
    atomic_json(root / 'learner.json', artifact['learner'])
    atomic_json(root / 'reference.json', artifact['reference'])
    atomic_json(root / 'metadata.json', artifact['metadata'])
    manifest = {'schema_version': SCHEMA_VERSION, 'task': artifact['task'],
                'group_id': group_id, 'split': split, 'split_seed': split_seed,
                'files': {name: checksum(root / name) for name in ['learner.json', 'reference.json', 'metadata.json']}}
    atomic_json(root / 'manifest.json', manifest)
    return manifest


def verify_export(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if set(manifest['files']) != {'learner.json', 'reference.json', 'metadata.json'}:
        raise ValueError('Unexpected task artifact set')
    for name, expected in manifest['files'].items():
        if checksum(root / name) != expected:
            raise ValueError('Task artifact hash mismatch: ' + name)
    return manifest


def future_participation(execution, config):
    """Predict future event-object participation within a declared existing catalog.

    Query time is supplied by the recipe, never chosen using a future outcome.
    Future events involving newly born entities are outside this task's universe.
    """
    cutoff, horizon = float(config['cutoff']), float(config['horizon'])
    if not math.isfinite(cutoff) or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('A finite cutoff and positive horizon are required')
    entities = {k: v for k, v in execution['entities'].items() if v['created_at'] <= cutoff}
    history = [e for e in execution['events'] if e['available_at'] <= cutoff]
    targets = [e for e in execution['events'] if cutoff < e['time'] <= cutoff + horizon
               and all(o in entities for o in e['participants'])]
    return {'task': 'future_participation',
            'learner': {'entities': entities, 'events': history, 'schema': execution['schema'],
                        'query': {'cutoff': cutoff, 'horizon': horizon, 'catalog_policy': 'existing_entities'}},
            'reference': {'future_events': targets},
            'metadata': {'world_type': execution['world_type'],
                         'excluded_future_entity_events': sum(cutoff < e['time'] <= cutoff + horizon
                                                             and any(o not in entities for o in e['participants'])
                                                             for e in execution['events']),
                         'provenance': execution['provenance']}}
