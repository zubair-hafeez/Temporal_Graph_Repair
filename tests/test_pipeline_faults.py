from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.factory_plugin import registry,learner_graph
from tgrepair.pipeline_faults import pipeline_faults
from tgrepair.edits import RepairState,teacher_edits
from tgrepair.metrics import traceability


def test_pipeline_faults_preserve_records_and_have_exact_private_labels():
    modules=registry();execution=modules.create_world('workflow',{'jobs':12,'seed':24,'work_stages':6,'lifecycle_records':True}).run()
    roles=execution['schema']['entity_roles'];types={roles['parent'],roles['child']}
    clean=modules.build_task('temporal_graph_repair',execution,{'target_types':sorted(types),'observation_seed':0,
                                                             'missing_rate':0,'wrong_rate':0,'fragment_rate':0})
    original=learner_graph(clean['learner'])
    observed,reference=pipeline_faults(original,31,types,outage_fraction=.15,stale_entity_fraction=.4,namespace_fraction=.8)
    observed.validate();assert reference['wrong'] and reference['missing'] and reference['fragmentations']
    originals={e.uid:e for e in original.events}
    for event in observed.events:
        source=originals[reference['event_origin'][event.uid]]
        assert (event.time,event.kind,event.attributes)==(source.time,source.kind,source.attributes)
    state=RepairState(observed)
    for edit in teacher_edits(state,reference,types):state.apply(edit)
    assert traceability(state.graph,reference,types,state.clusters)['f1']==1.
    again,other=pipeline_faults(original,31,types,outage_fraction=.15,stale_entity_fraction=.4,namespace_fraction=.8)
    assert observed==again and reference==other
