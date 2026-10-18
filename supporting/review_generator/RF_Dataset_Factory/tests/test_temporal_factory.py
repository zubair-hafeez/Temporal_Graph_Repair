"""Acceptance for the paper-bounded temporal extension; no RF engine runs."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from temporal_factory.core import (
    Registry, SCHEMA_VERSION, export_task, fingerprint, future_participation,
    validate_execution, verify_export,
)
from temporal_factory.workflow import WorkflowWorld, register


def test_json_checkpoint_resumes_exactly():
    for seed in range(3):
        world = WorkflowWorld({'seed': seed, 'jobs': 8})
        world.advance(31)
        checkpoint = json.loads(json.dumps(world.checkpoint()))
        assert fingerprint(world.run()) == fingerprint(WorkflowWorld.restore(checkpoint).run())


def test_checkpoint_rejects_changed_source_and_recipe():
    checkpoint = WorkflowWorld({'jobs': 2}).checkpoint()
    checkpoint['source_hash'] = 'incompatible'
    with pytest.raises(ValueError, match='source'):
        WorkflowWorld.restore(checkpoint)
    checkpoint = WorkflowWorld({'jobs': 2}).checkpoint()
    checkpoint['recipe']['jobs'] = 3
    with pytest.raises(ValueError, match='recipe'):
        WorkflowWorld.restore(checkpoint)


def test_resource_queues_are_nonoverlapping():
    result = WorkflowWorld({'jobs': 16, 'resources': 2, 'arrival_interval': .1}).run()
    by_resource = {}
    for interval in result['reference']['service_intervals']:
        by_resource.setdefault(interval['resource'], []).append(interval)
    for intervals in by_resource.values():
        ordered = sorted(intervals, key=lambda x: x['start'])
        assert all(a['finish'] <= b['start'] for a, b in zip(ordered, ordered[1:]))


def test_mechanisms_have_observable_effects_and_exceptions_survive():
    base = {'jobs': 20, 'seed': 2, 'retry_probability': 1, 'cancel_probability': 0, 'return_probability': 1}
    enabled = WorkflowWorld(base).run()
    disabled = WorkflowWorld({**base, 'mechanisms': []}).run()
    types = {e['kind'] for e in enabled['events']}
    assert {'Review A', 'Review B', 'Check failed', 'Return', 'Consolidate WorkBatch'} <= types
    assert not {'Review A', 'Review B', 'Check failed', 'Return', 'Consolidate WorkBatch'} & {e['kind'] for e in disabled['events']}
    assert enabled['reference']['exception_events']
    assert all(eid in {e['id'] for e in enabled['events']} for eid in enabled['reference']['exception_events'])
    canceled = WorkflowWorld({'jobs': 10, 'cancel_probability': 1, 'retry_probability': 0}).run()
    assert 'Cancel' in {e['kind'] for e in canceled['events']}
    assert 'Complete' not in {e['kind'] for e in canceled['events']}


def independent_spatial_execution():
    return {'schema_version': SCHEMA_VERSION, 'world_type': 'spatial_fixture',
            'entities': {'room': {'kind': 'Room', 'created_at': 0., 'attributes': {'size_m': [4., 5., 3.]}},
                         'robot': {'kind': 'Robot', 'created_at': 0., 'attributes': {'initial_position_m': [0., 0., 0.]}},
                         'late': {'kind': 'Robot', 'created_at': 9., 'attributes': {}}},
            'events': [{'id': 'e0', 'time': 1., 'available_at': 1., 'kind': 'Enter', 'participants': ['room', 'robot'], 'attributes': {}},
                       {'id': 'delayed', 'time': 2., 'available_at': 7., 'kind': 'Inspect', 'participants': ['room', 'robot'], 'attributes': {}},
                       {'id': 'next', 'time': 6., 'available_at': 6., 'kind': 'Leave', 'participants': ['room', 'robot'], 'attributes': {}},
                       {'id': 'born_later', 'time': 9., 'available_at': 9., 'kind': 'Enter', 'participants': ['room', 'late'], 'attributes': {}}],
            'schema': {'entity_types': ['Room', 'Robot'], 'event_types': ['Enter', 'Inspect', 'Leave']},
            'capabilities': ['event_history', 'entity_catalog'], 'provenance': {'fidelity': 'contract_fixture'}}


def test_prefix_task_respects_availability_and_future_entities():
    execution = independent_spatial_execution()
    validate_execution(execution)
    artifact = future_participation(execution, {'cutoff': 4., 'horizon': 8.})
    assert {e['id'] for e in artifact['learner']['events']} == {'e0'}
    assert set(artifact['learner']['entities']) == {'room', 'robot'}
    assert {e['id'] for e in artifact['reference']['future_events']} == {'next'}
    assert artifact['metadata']['excluded_future_entity_events'] == 1


def test_registry_rejects_unsupported_capability_and_private_fields():
    registry = Registry()
    register(registry)
    execution = registry.create_world('workflow', {'jobs': 2}).run()
    registry.register_task('intervention', lambda e, c: None, {'counterfactual_interventions'})
    with pytest.raises(ValueError, match='unsupported'):
        registry.build_task('intervention', execution, {})
    registry.register_task('leaky', lambda e, c: {'task': 'leaky', 'learner': {'truth': e}, 'reference': {}, 'metadata': {}}, set())
    with pytest.raises(ValueError, match='Evaluator-only'):
        registry.build_task('leaky', execution, {})


def test_export_hashes_and_split_ancestry(tmp_path):
    registry = Registry(); register(registry)
    execution = registry.create_world('workflow', {'jobs': 3}).run()
    artifact = registry.build_task('future_participation', execution, {'cutoff': 5., 'horizon': 10.})
    a = export_task(tmp_path / 'a', artifact, 'same-execution')
    b = export_task(tmp_path / 'b', artifact, 'same-execution')
    assert a == b
    assert verify_export(tmp_path / 'a') == a
    (tmp_path / 'a' / 'learner.json').write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_export(tmp_path / 'a')


def test_rejects_dangling_entities_and_nonfinite_attributes():
    execution = independent_spatial_execution()
    execution['events'][0]['participants'].append('missing')
    with pytest.raises(ValueError, match='Unknown'):
        validate_execution(execution)
    execution = independent_spatial_execution()
    execution['entities']['room']['attributes']['size_m'][0] = float('nan')
    with pytest.raises(ValueError):
        validate_execution(execution)


def test_workflow_does_not_import_radio_backend():
    WorkflowWorld({'jobs': 2, 'family': 'assembly'}).run()
    assert not any(name.startswith(('sionna', 'mitsuba', 'drjit')) for name in sys.modules)


def test_long_routes_lifecycle_and_waiting_are_operative_and_resumable():
    recipe = {'jobs': 8, 'seed': 3, 'resources': 2, 'work_stages': 12,
              'work_repetitions': 3, 'lifecycle_records': True, 'waiting_scale': 20.}
    world = WorkflowWorld(recipe)
    world.advance(91)
    checkpoint = json.loads(json.dumps(world.checkpoint()))
    result = world.run()
    assert fingerprint(result) == fingerprint(WorkflowWorld.restore(checkpoint).run())
    base = WorkflowWorld({'jobs': 8, 'seed': 3, 'resources': 2}).run()
    assert len(result['events']) > 2 * len(base['events'])
    assert any(e['kind'].startswith('Work ') for e in result['events'])
    assert any(e['kind'].endswith('[start]') for e in result['events'])
    assert result['events'][-1]['time'] > base['events'][-1]['time']
    for resource in {s['resource'] for s in result['reference']['service_intervals']}:
        intervals=sorted([s for s in result['reference']['service_intervals'] if s['resource']==resource],key=lambda s:s['start'])
        assert all(a['finish']<=b['start'] for a,b in zip(intervals,intervals[1:]))
    assert {e['kind'] for e in result['events']} <= set(result['schema']['event_types'])
    other = WorkflowWorld({**recipe, 'seed': 17}).run()
    assert result['schema']['event_types'] == other['schema']['event_types']
