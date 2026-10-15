from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.data import Graph,Event
from tgrepair.edits import RepairState,Edit
from tgrepair.application_queries import audit_queries


def test_fragmented_history_hides_cancellation_evidence_from_one_handle():
    graph=Graph('fixture',{'a':'Case','alias':'Case'},[Event('e0',1.,'Open',('a',)),Event('e1',2.,'Cancel',('alias',))])
    reference={'identity':{'a':'a','alias':'a'},'event_participation':{'e0':['a'],'e1':['a']}}
    before=audit_queries(graph,reference,{'Case'})
    assert before['queries']['cancellation_evidence']['recall']==.5
    state=RepairState(graph);state.apply(Edit('MERGE_ALIASES','a','alias'))
    after=audit_queries(state.graph,reference,{'Case'},state.clusters)
    assert after['queries']['cancellation_evidence']['f1']==1.
    assert after['first_timestamp_correct_fraction']==after['last_timestamp_correct_fraction']==1.
