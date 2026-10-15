"""Research repair task registered with the shared temporal dataset factory."""
from dataclasses import asdict
from pathlib import Path
import sys

FACTORY_SOURCE = Path(__file__).resolve().parents[2] / 'RF_Dataset_Factory/src'
if str(FACTORY_SOURCE) not in sys.path:
    sys.path.insert(0, str(FACTORY_SOURCE))

from temporal_factory.core import Registry
from temporal_factory.workflow import register as register_workflow
from .data import Event, Graph, sha256
from .corruption import mixed_faults


def repair_task(execution, config):
    if not execution['provenance'].get('complete', False):
        raise ValueError('Repair labels require a completed reference execution')
    target_types = set(config['target_types'])
    if not target_types <= set(execution['schema']['entity_types']):
        raise ValueError('Unknown repair target type')
    visible_types = set(config.get('visible_types', execution['schema']['entity_types']))
    if not target_types <= visible_types or not visible_types <= set(execution['schema']['entity_types']):
        raise ValueError('Repair targets must belong to the declared visible entity types')
    objects = {o:e['kind'] for o,e in execution['entities'].items() if e['kind'] in visible_types}
    events = []
    for e in execution['events']:
        participants = tuple(o for o in e['participants'] if o in objects)
        if participants:
            events.append(Event(e['id'],e['time'],e['kind'],participants,dict(e['attributes'])))
    graph = Graph('workflow_reference', objects, events)
    observed, reference = mixed_faults(graph,config['observation_seed'],target_types,
                                       missing_rate=config.get('missing_rate',.1),wrong_rate=config.get('wrong_rate',.1),
                                       fragment_rate=config.get('fragment_rate',.25),gap_mode=config.get('gap_mode','contiguous'))
    exceptional = set(execution['reference'].get('exception_events', []))
    reference['exception_events'] = [eid for eid, original in reference['event_origin'].items() if original in exceptional]
    learner = asdict(observed)
    learner['schema'] = {**execution['schema'], 'entity_types':sorted(visible_types),
                         'entity_roles':{role:typ for role,typ in execution['schema'].get('entity_roles',{}).items() if typ in visible_types},
                         'target_types':sorted(target_types),
                         'protected_fields':['time','kind','attributes'],
                         'allowed_edits':['ADD_PARTICIPANT','REMOVE_PARTICIPANT','MERGE_ALIASES','STOP','UNRESOLVED']}
    return {'task':'temporal_graph_repair','learner':learner,'reference':reference,
            'metadata':{'world_type':execution['world_type'],'provenance':execution['provenance'],
                        'observation_policy':dict(config),'task_plugin_hash':sha256(__file__),
                        'reference_quality':'exact_symbolic_execution',
                        'projection_policy':'retain events involving at least one declared visible entity type, then corrupt observations',
                        'observation_window':'complete_offline_history',
                        'entity_catalog':'available_registry_with_fragmented_handles'}}


def registry():
    result=Registry()
    register_workflow(result)
    result.register_task('temporal_graph_repair',repair_task,{'event_history','entity_catalog'})
    return result


def learner_graph(learner):
    allowed={'name','objects','events','schema'}
    if set(learner)!=allowed:
        raise ValueError('Unexpected learner fields; check evaluator separation')
    result=Graph(learner['name'],learner['objects'],
                 [Event(e['uid'],e['time'],e['kind'],tuple(e['participants']),e.get('attributes',{}))
                  for e in learner['events']])
    result.validate()
    return result
