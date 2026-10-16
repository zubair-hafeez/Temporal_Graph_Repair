from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.data import Event,Graph
from tgrepair.view_decoding import decode_with_view


def test_view_edit_replay_and_undo_preserve_native_fields_without_reference():
    graph=Graph('fixture',{'a':'Case','b':'Case','r':'Resource'},[
        Event('e0',0.,'Start',('a','r'),{'native_note':'keep start'}),
        Event('e1',1.,'Work',('r',),{'native_note':'keep work'}),
        Event('e2',2.,'Finish',('a','r'),{'native_note':'keep finish'})])
    def scorer(observation):
        assert all(not event.attributes for event in observation.graph.events)
        return np.asarray([1. if edit.operation=='ADD_PARTICIPANT' and edit.source=='e1' and edit.target=='a' else 0.
                           for edit in observation.edits])
    state,audit=decode_with_view(graph,{'Case'},scorer,[.9,1.1,1.1],attribute_fields=[],rounds=1)
    assert state.graph.events[1].participants==('a','r')
    assert [e.attributes for e in state.graph.events]==[e.attributes for e in graph.events]
    assert audit['native_event_fields_preserved']
    state.undo()
    assert state.graph==graph
