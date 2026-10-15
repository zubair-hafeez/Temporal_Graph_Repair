from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.factory_plugin import learner_graph, registry
from temporal_factory.core import export_task, verify_export


def test_same_execution_supports_repair_and_prediction(tmp_path):
    modules=registry()
    execution=modules.create_world('workflow',{'jobs':6,'seed':4,'retry_probability':1,'return_probability':1,'cancel_probability':0}).run()
    roles=execution['schema']['entity_roles']
    repair=modules.build_task('temporal_graph_repair',execution,
                              {'target_types':[roles['parent'],roles['child']],'observation_seed':31})
    prediction=modules.build_task('future_participation',execution,{'cutoff':8,'horizon':30})
    assert repair['metadata']['execution_sha256']==prediction['metadata']['execution_sha256']
    graph=learner_graph(repair['learner'])
    assert len(graph.events)==len(execution['events'])
    assert len(repair['reference']['exception_events'])==len(execution['reference']['exception_events'])
    assert not {'identity','event_origin','faults','recipe','seed'} & set(repair['learner'])
    assert all(e['available_at']<=8 for e in prediction['learner']['events'])
    a=export_task(tmp_path/'repair',repair,'same-world')
    b=export_task(tmp_path/'prediction',prediction,'same-world')
    assert a['split']==b['split']
    assert verify_export(tmp_path/'repair')==a


def test_clean_variant_preserves_exceptions_exactly():
    modules=registry()
    execution=modules.create_world('workflow',{'jobs':4,'seed':7,'cancel_probability':1,'retry_probability':0}).run()
    roles=execution['schema']['entity_roles']
    artifact=modules.build_task('temporal_graph_repair',execution,
                               {'target_types':[roles['child']],'observation_seed':12,
                                'missing_rate':0,'wrong_rate':0,'fragment_rate':0})
    graph=learner_graph(artifact['learner'])
    reference=artifact['reference']
    assert reference['exception_events']
    assert not reference['wrong'] and not reference['missing'] and not reference['fragmentations']
    for e in graph.events:
        assert set(e.participants)==set(reference['event_participation'][e.uid])


def test_observation_seed_does_not_change_reference_behavior():
    modules=registry()
    execution=modules.create_world('workflow',{'jobs':6,'seed':2}).run()
    roles=execution['schema']['entity_roles']
    outputs=[modules.build_task('temporal_graph_repair',execution,
                                {'target_types':[roles['parent'],roles['child']],'observation_seed':seed})
             for seed in [5,17]]
    assert outputs[0]['metadata']['execution_sha256']==outputs[1]['metadata']['execution_sha256']
    for output in outputs:
        original={e['id']:(e['time'],e['kind'],e['attributes']) for e in execution['events']}
        for e in learner_graph(output['learner']).events:
            assert (e.time,e.kind,e.attributes)==original[output['reference']['event_origin'][e.uid]]


def test_flat_projection_changes_observation_and_not_execution():
    modules=registry()
    execution=modules.create_world('workflow',{'jobs':4,'seed':0}).run()
    roles=execution['schema']['entity_roles']
    artifact=modules.build_task('temporal_graph_repair',execution,
                               {'target_types':[roles['child']], 'visible_types':[roles['child'],roles['resource']],
                                'observation_seed':11})
    graph=learner_graph(artifact['learner'])
    assert set(graph.objects.values())=={roles['child'],roles['resource']}
    assert roles['parent'] in {v['kind'] for v in execution['entities'].values()}
    assert len(graph.events)<len(execution['events'])


def test_attribute_projection_is_observation_only():
    modules=registry();execution=modules.create_world('workflow',{'jobs':4,'seed':4}).run()
    roles=execution['schema']['entity_roles']
    artifact=modules.build_task('temporal_graph_repair',execution,
                                {'target_types':[roles['child']],'visible_attributes':['Amount'],'observation_seed':1})
    assert set(artifact['learner']['schema']['attributes'])=={'Amount'}
    assert all(set(e.attributes)<={'Amount'} for e in learner_graph(artifact['learner']).events)
    assert any('Priority' in e['attributes'] for e in execution['events'])
