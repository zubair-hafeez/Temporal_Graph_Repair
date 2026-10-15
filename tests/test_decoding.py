from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.data import Event,Graph
from tgrepair.decoding import decode


def test_decoder_discovers_missing_participant_and_stops_without_reference():
    graph=Graph('observed',{'a':'Case','r':'Resource'},[
        Event('e1',1.,'Open',('a','r')),
        Event('e2',2.,'Finish',('r',)),
    ])
    def scorer(observation):
        return np.asarray([.99 if e.operation=='ADD_PARTICIPANT' and e.source=='e2' and e.target=='a' else 0. for e in observation.edits])
    state,audit=decode(graph,{'Case'},scorer,[.95,.95,.95])
    assert state.graph.events[1].participants==('a','r')
    assert audit['edit_count']==1 and audit['stopped']
    assert graph.events[1].participants==('r',)


def test_decoder_abstains_on_unsupported_actions():
    graph=Graph('observed',{'a':'Case','b':'Case'},[Event('e1',1.,'Open',('a',)),Event('e2',2.,'Open',('b',))])
    state,audit=decode(graph,{'Case'},lambda observation:np.zeros(len(observation.edits)),[.95,.95,.95])
    assert not state.log and audit['stopped']
    assert state.graph.events==graph.events


def test_round_checkpoint_replay_matches_uninterrupted_decoding():
    graph=Graph('observed',{'a':'Case','r':'Resource'},[
        Event('e1',1.,'Open',('a','r')),Event('e2',2.,'Finish',('r',))])
    def scorer(observation):
        return np.asarray([.99 if e.operation=='ADD_PARTICIPANT' and e.source=='e2' else 0. for e in observation.edits])
    snapshots=[]
    partial,part_audit=decode(graph,{'Case'},scorer,[.95]*3,rounds=1,checkpoint=lambda history:snapshots.append(list(history)))
    resumed,resumed_audit=decode(graph,{'Case'},scorer,[.95]*3,rounds=8,resume_history=snapshots[0])
    complete,complete_audit=decode(graph,{'Case'},scorer,[.95]*3,rounds=8)
    assert resumed.graph==complete.graph and resumed.clusters==complete.clusters
    assert resumed_audit['rounds']==complete_audit['rounds'] and resumed_audit['edit_count']==1
