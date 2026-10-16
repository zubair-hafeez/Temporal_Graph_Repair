"""Optional matching views with repairs replayed on immutable native records."""
from .attribute_views import stable_attribute_view
from .data import Event,Graph
from .decoding import decode
from .edits import Edit,RepairState
from .action_features import propose_actions


def select_fields(graph,target_types,mode='all',threshold=.5):
    if mode=='all':return None,{'mode':'all'}
    if mode=='none':return [],{'mode':'none','retained_fields':[]}
    if mode!='stable':raise ValueError(mode)
    _,profile=stable_attribute_view(graph,target_types,threshold)
    return profile['retained_fields'],{'mode':'stable',**profile,
                                      'selection_context':'Initial observed graph only; fixed through decoding'}


def propose_in_view(state,target_types,attribute_fields=None,**options):
    if attribute_fields is None:return propose_actions(state,target_types,**options)
    graph=state.active_graph();fields=set(attribute_fields)
    view=Graph(graph.name,dict(graph.objects),[
        Event(e.uid,e.time,e.kind,e.participants,{k:v for k,v in e.attributes.items() if k in fields})
        for e in graph.events])
    return propose_actions(RepairState(view),target_types,**options)


def decode_with_view(graph,target_types,scorer,thresholds,attribute_fields=None,**options):
    if attribute_fields is None:return decode(graph,target_types,scorer,thresholds,**options)
    fields=set(attribute_fields)
    view=Graph(graph.name,dict(graph.objects),[
        Event(e.uid,e.time,e.kind,e.participants,{k:v for k,v in e.attributes.items() if k in fields})
        for e in graph.events])
    view_state,audit=decode(view,target_types,scorer,thresholds,**options)
    # Replay public edit decisions on the unmodified native observation. All undo
    # states then also retain native descriptions; no evaluation labels needed.
    state=RepairState(graph)
    for record in audit['rounds']:
        state.apply_batch([Edit(**item['edit']) for item in record['edits']])
    if state.clusters!=view_state.clusters:raise AssertionError('View replay changed identity edits')
    if [(e.uid,e.participants) for e in state.graph.events]!=[(e.uid,e.participants) for e in view_state.graph.events]:
        raise AssertionError('View replay changed participation edits')
    audit['matching_attribute_fields']=sorted(fields)
    audit['native_event_fields_preserved']=True
    return state,audit
