"""Small executable workflow worlds for temporal repair research.

This is a symbolic simulator with declared mechanisms, not a calibrated model
of a particular company. Faults are applied by observation/task plugins later.
"""
from __future__ import annotations

import copy
import heapq
import math
from pathlib import Path
import random
import sys

from rfdf.storage import checksum
from . import core
from .core import SCHEMA_VERSION, fingerprint, validate_execution


MECHANISMS = {'branch_join', 'batch_split', 'handoff', 'retry', 'cancel_return', 'queues'}
FAMILIES = {
    'service': ('Request', 'Case', 'Agent', 'WorkBatch'),
    'fulfillment': ('Order', 'Item', 'Worker', 'Container'),
    'assembly': ('WorkOrder', 'Component', 'Machine', 'Lot'),
}
CAPABILITIES = {'event_history', 'entity_catalog', 'stateful_resume'}
DEFAULTS = {'version': 1, 'seed': 0, 'family': 'service', 'jobs': 12, 'max_children': 3,
            'resources': 4, 'arrival_interval': 3.0, 'service_scale': 1.0,
            'batch_window': 5.0, 'mechanisms': sorted(MECHANISMS),
            'retry_probability': .2, 'cancel_probability': .1, 'return_probability': .1,
            'attribute_palette': 8, 'work_stages': 0, 'work_repetitions': 1,
            'lifecycle_records': False, 'waiting_scale': 1.0}


def normalize_recipe(recipe):
    unknown = set(recipe) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown workflow recipe fields: ' + repr(sorted(unknown)))
    r = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(recipe)}
    if r['version'] != 1 or r['family'] not in FAMILIES:
        raise ValueError('Unsupported workflow recipe version or family')
    for key, lower, upper in [('jobs', 1, 256), ('max_children', 1, 8), ('resources', 1, 64), ('attribute_palette', 1, 1000),
                              ('work_stages', 0, 24), ('work_repetitions', 1, 4)]:
        if type(r[key]) is not int or not lower <= r[key] <= upper:
            raise ValueError('Invalid bounded integer: ' + key)
    if type(r['seed']) is not int:
        raise ValueError('Seed must be an integer')
    for key in ['arrival_interval', 'service_scale', 'batch_window', 'waiting_scale']:
        if not isinstance(r[key], (int, float)) or not math.isfinite(r[key]) or r[key] <= 0:
            raise ValueError('Invalid positive time scale: ' + key)
    for key in ['retry_probability', 'cancel_probability', 'return_probability']:
        if not isinstance(r[key], (int, float)) or not 0 <= r[key] <= 1:
            raise ValueError('Invalid probability: ' + key)
    if len(set(r['mechanisms'])) != len(r['mechanisms']) or not set(r['mechanisms']) <= MECHANISMS:
        raise ValueError('Unknown or duplicate mechanism')
    r['mechanisms'] = sorted(r['mechanisms'])
    if type(r['lifecycle_records']) is not bool:
        raise ValueError('Lifecycle record option must be boolean')
    core.canonical_bytes(r)
    return r


def source_fingerprint():
    return fingerprint({'workflow': checksum(Path(__file__)), 'core': checksum(Path(core.__file__)),
                        'python': list(sys.version_info[:2])})


def tuple_tree(value):
    return tuple(tuple_tree(x) for x in value) if isinstance(value, (tuple, list)) else value


class WorkflowWorld:
    def __init__(self, recipe):
        self.recipe = normalize_recipe(recipe)
        self.recipe_hash = fingerprint(self.recipe)
        self.source_hash = source_fingerprint()
        self.rng = random.Random(self.recipe['seed'])
        self.id_rng = random.Random(self.recipe['seed'] + 98371)
        self.entities = {}
        self.events = []
        self.pending = []
        self.sequence = 0
        self.action_count = 0
        self.items = {}
        self.batches = {}
        self.resources = []
        self.resource_free = {}
        self.service_intervals = []
        self.exception_events = []
        self.mechanisms = set(self.recipe['mechanisms'])
        parent_kind, child_kind, resource_kind, batch_kind = FAMILIES[self.recipe['family']]
        self.kinds = {'parent': parent_kind, 'child': child_kind, 'resource': resource_kind, 'batch': batch_kind}
        for _ in range(self.recipe['resources']):
            uid = self.new_entity(resource_kind, 0.0, {'capacity': 1})
            self.resources.append(uid)
            self.resource_free[uid] = 0.0
        for i in range(self.recipe['jobs']):
            when = (i + self.rng.random()) * self.recipe['arrival_interval']
            self.schedule(when, 'arrival', {})

    def uid(self, prefix):
        return prefix + f'{self.id_rng.getrandbits(96):024x}'

    def new_entity(self, kind, time, attributes):
        uid = self.uid('n')
        if uid in self.entities:
            raise RuntimeError('Unexpected identifier collision')
        self.entities[uid] = {'kind': kind, 'created_at': time, 'attributes': dict(attributes)}
        return uid

    def schedule(self, time, action, payload):
        self.sequence += 1
        heapq.heappush(self.pending, (time, self.sequence, action, payload))

    def emit(self, time, kind, participants, attributes=None, exception=False):
        event = {'id': self.uid('e'), 'time': time, 'available_at': time, 'kind': kind,
                 'participants': sorted(set(participants)), 'attributes': dict(attributes or {})}
        self.events.append(event)
        if exception:
            self.exception_events.append(event['id'])

    def request(self, time, item, phase):
        self.schedule(time, 'request', {'item': item, 'phase': phase})

    def step(self):
        if not self.pending:
            return False
        time, _, action, payload = heapq.heappop(self.pending)
        self.action_count += 1
        if self.action_count > 500000:
            raise RuntimeError('Bounded workflow action budget exceeded')
        if action == 'arrival':
            attrs = {'Amount': 100 * self.rng.randint(1, self.recipe['attribute_palette']),
                     'Priority': self.rng.choice(['normal', 'urgent']),
                     'Category': self.rng.choice(['a', 'b', 'c'])}
            parent = self.new_entity(self.kinds['parent'], time, attrs)
            self.emit(time, 'Register ' + self.kinds['parent'], [parent], attrs)
            for _ in range(self.rng.randint(1, self.recipe['max_children'])):
                child = self.new_entity(self.kinds['child'], time, attrs)
                self.items[child] = {'parent': parent, 'attributes': dict(attrs), 'attempts': 0,
                                     'gates': 2 if 'branch_join' in self.mechanisms else 0,
                                     'worker': self.rng.choice(self.resources), 'batch': None, 'terminal': False}
                stages = self.recipe['work_stages']
                # A per-case finite work route creates legitimate long histories.
                # Extra randomness is drawn only when this option is enabled.
                self.items[child]['route'] = ([f'Work {stage:02d}' for stage in range(1, self.rng.randint(1, stages) + 1)]
                                              * self.rng.randint(1, self.recipe['work_repetitions'])) if stages else []
                self.items[child]['route_position'] = 0
                self.emit(time, 'Open ' + self.kinds['child'], [parent, child], attrs)
                if 'branch_join' in self.mechanisms:
                    self.request(time, child, 'Review A')
                    self.request(time, child, 'Review B')
                else:
                    self.request(time, child, 'Prepare')
        elif action == 'request':
            item, phase = payload['item'], payload['phase']
            state = self.items[item]
            worker = self.rng.choice(self.resources) if 'handoff' in self.mechanisms else state['worker']
            duration = self.recipe['service_scale'] * self.rng.lognormvariate(0, .7)
            if 'queues' in self.mechanisms:
                start = max(time, self.resource_free[worker])
                finish = start + duration
                self.resource_free[worker] = finish
            else:
                start, finish = time, time + duration
            self.service_intervals.append({'resource': worker, 'start': start, 'finish': finish,
                                           'capacity_enforced': 'queues' in self.mechanisms})
            if self.recipe['lifecycle_records']:
                participants = [item, state['parent'], worker]
                if state['batch']:
                    participants.append(state['batch'])
                self.emit(start, phase + ' [start]', participants, state['attributes'])
            self.schedule(finish, 'finish', {'item': item, 'phase': phase, 'worker': worker})
        elif action == 'finish':
            item, phase, worker = payload['item'], payload['phase'], payload['worker']
            state = self.items[item]
            participants = [item, state['parent'], worker]
            if state['batch']:
                participants.append(state['batch'])
            kind, exceptional = phase, False
            outcome = None
            if phase == 'Check':
                if 'retry' in self.mechanisms and state['attempts'] < 2 and self.rng.random() < self.recipe['retry_probability']:
                    outcome, kind, exceptional = 'retry', 'Check failed', True
                elif 'cancel_return' in self.mechanisms and self.rng.random() < self.recipe['cancel_probability']:
                    outcome, kind, exceptional = 'cancel', 'Cancel', True
            if phase == 'Return':
                exceptional = True
            self.emit(time, kind + (' [complete]' if self.recipe['lifecycle_records'] else ''), participants, state['attributes'], exceptional)
            delay = self.rng.expovariate(1 / self.recipe['service_scale']) * self.recipe['waiting_scale']
            if phase in {'Review A', 'Review B'}:
                state['gates'] -= 1
                if state['gates'] == 0:
                    self.request(time + delay, item, 'Prepare')
            elif phase == 'Prepare':
                if 'batch_split' in self.mechanisms:
                    bucket = int(time // self.recipe['batch_window']) + 1
                    key = str(bucket)
                    if key not in self.batches:
                        self.batches[key] = []
                        self.schedule(bucket * self.recipe['batch_window'], 'flush_batch', {'key': key})
                    self.batches[key].append(item)
                else:
                    self.start_route(time + delay, item)
            elif phase.startswith('Work '):
                self.start_route(time + delay, item)
            elif phase == 'Check':
                if outcome == 'retry':
                    state['attempts'] += 1
                    self.request(time + delay, item, 'Check')
                elif outcome == 'cancel':
                    state['terminal'] = True
                else:
                    self.request(time + delay, item, 'Complete')
            elif phase == 'Complete':
                if 'cancel_return' in self.mechanisms and self.rng.random() < self.recipe['return_probability']:
                    self.request(time + 5 * delay, item, 'Return')
                else:
                    state['terminal'] = True
            elif phase == 'Return':
                state['terminal'] = True
        elif action == 'flush_batch':
            children = self.batches.pop(payload['key'])
            batch = self.new_entity(self.kinds['batch'], time, {'size': len(children)})
            parents = [self.items[item]['parent'] for item in children]
            self.emit(time, 'Consolidate ' + self.kinds['batch'], children + parents + [batch])
            for item in children:
                self.items[item]['batch'] = batch
                self.start_route(time, item)
        else:
            raise ValueError('Unknown pending action: ' + action)
        return True

    def start_route(self, time, item):
        state = self.items[item]
        position = state['route_position']
        if position < len(state['route']):
            state['route_position'] += 1
            self.request(time, item, state['route'][position])
        else:
            self.request(time, item, 'Check')

    def advance(self, actions):
        if type(actions) is not int or actions < 0:
            raise ValueError('Nonnegative bounded action count required')
        for _ in range(actions):
            if not self.step():
                break
        return not self.pending

    def run(self):
        while self.step():
            pass
        if not all(item['terminal'] for item in self.items.values()):
            raise RuntimeError('World ended with unfinished work')
        return self.execution()

    def execution(self):
        events = sorted(copy.deepcopy(self.events), key=lambda e: (e['time'], e['id']))
        record_types = ['Register ' + self.kinds['parent'], 'Open ' + self.kinds['child'], 'Consolidate ' + self.kinds['batch']]
        phases = ['Review A', 'Review B', 'Prepare', 'Check', 'Check failed', 'Cancel', 'Complete', 'Return']
        phases += [f'Work {stage:02d}' for stage in range(1, self.recipe['work_stages'] + 1)]
        # The declared vocabulary is recipe-derived, never selected from future outcomes.
        record_types += ([phase + suffix for phase in phases for suffix in [' [start]', ' [complete]']]
                         if self.recipe['lifecycle_records'] else phases)
        execution = {'schema_version': SCHEMA_VERSION, 'world_type': 'workflow',
                     'entities': copy.deepcopy(self.entities), 'events': events,
                     'schema': {'entity_types': sorted(self.kinds.values()),
                                'entity_roles': dict(self.kinds),
                                'event_types': record_types,
                                'attributes': {'Amount': 'numeric', 'Priority': 'categorical', 'Category': 'categorical'}},
                     'capabilities': sorted(CAPABILITIES),
                     'provenance': {'recipe': copy.deepcopy(self.recipe), 'recipe_hash': self.recipe_hash,
                                    'source_hash': self.source_hash, 'fidelity': 'symbolic_workflow',
                                    'mechanism_family': fingerprint({'family': self.recipe['family'],
                                                                     'mechanisms': self.recipe['mechanisms']}),
                                    'complete': not self.pending},
                     'reference': {'exception_events': list(self.exception_events),
                                   'service_intervals': copy.deepcopy(self.service_intervals)}}
        return validate_execution(execution)

    def checkpoint(self):
        return copy.deepcopy({'checkpoint_version': 1, 'recipe': self.recipe, 'recipe_hash': self.recipe_hash,
                              'source_hash': self.source_hash, 'rng': self.rng.getstate(), 'id_rng': self.id_rng.getstate(),
                              **{name: getattr(self, name) for name in ['entities', 'events', 'pending', 'sequence',
                                  'action_count', 'items', 'batches', 'resources', 'resource_free',
                                  'service_intervals', 'exception_events']}})

    @classmethod
    def restore(cls, checkpoint):
        if checkpoint['checkpoint_version'] != 1 or checkpoint['source_hash'] != source_fingerprint():
            raise ValueError('Checkpoint source/version mismatch')
        world = cls(checkpoint['recipe'])
        if world.recipe_hash != checkpoint['recipe_hash']:
            raise ValueError('Checkpoint recipe mismatch')
        world.rng.setstate(tuple_tree(checkpoint['rng']))
        world.id_rng.setstate(tuple_tree(checkpoint['id_rng']))
        for name in ['entities', 'events', 'pending', 'sequence', 'action_count', 'items', 'batches',
                     'resources', 'resource_free', 'service_intervals', 'exception_events']:
            setattr(world, name, copy.deepcopy(checkpoint[name]))
        world.pending = [tuple(entry) for entry in world.pending]
        heapq.heapify(world.pending)
        return world


def register(registry):
    registry.register_world('workflow', WorkflowWorld, CAPABILITIES)
    registry.register_task('future_participation', core.future_participation, {'event_history', 'entity_catalog'})
