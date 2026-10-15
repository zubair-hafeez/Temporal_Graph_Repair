from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.data import Graph,Event
from tgrepair.corruption import mixed_faults
from tgrepair.edits import RepairState,Edit,teacher_edits
from tgrepair.metrics import traceability,identity_pair_scores
from tgrepair.factory_plugin import registry,learner_graph


def test_merge_is_reversible_and_protected_fields_survive():
    graph=Graph('fixture',{'a':'Case','b':'Case','r':'Resource'},
                [Event('e0',0,'Cancel',('a','r'),{'amount':5}),Event('e1',1,'Retry',('b','r'),{'amount':5})])
    state=RepairState(graph)
    state.apply(Edit('MERGE_ALIASES','a','b'))
    assert state.clusters['a']==state.clusters['b']
    assert [(e.time,e.kind,e.attributes) for e in state.graph.events]==[(e.time,e.kind,e.attributes) for e in graph.events]
    state.undo()
    assert state.graph==graph
    assert state.clusters=={'a':'a','b':'b','r':'r'}


def test_teacher_program_restores_reference_modulo_identity_equivalence():
    modules=registry();execution=modules.create_world('workflow',{'jobs':6,'seed':9}).run()
    roles=execution['schema']['entity_roles'];types={roles['parent'],roles['child']}
    artifact=modules.build_task('temporal_graph_repair',execution,{'target_types':sorted(types),'observation_seed':17,
                                                                 'missing_rate':.2,'wrong_rate':.2,'fragment_rate':.5})
    graph=learner_graph(artifact['learner']);reference=artifact['reference']
    state=RepairState(graph)
    program=teacher_edits(state,reference,types)
    assert program
    for edit in program:state.apply(edit)
    assert traceability(state.graph,reference,types,state.clusters)['f1']==1
    assert identity_pair_scores(state.graph,reference,types,state.clusters)['f1']==1
    assert teacher_edits(state,reference,types)==[]
    for _ in program:state.undo()
    assert state.graph==graph
