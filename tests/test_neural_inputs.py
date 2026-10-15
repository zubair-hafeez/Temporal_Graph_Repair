from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tgrepair.action_features import propose_actions
from tgrepair.data import Event,Graph
from tgrepair.edits import RepairState
from tgrepair.neural_policy import GraphEditPolicy,observation_tensors


def test_candidate_generation_never_requires_reference_labels():
    graph=Graph('fixture',{'a':'Case','b':'Case','r':'Resource'},
                [Event('e1',0,'start',('a','r')),Event('e2',1,'end',('b','r'))])
    observed=propose_actions(RepairState(graph),{'Case'},event_budget=None)
    assert {e.operation for e in observed.edits}=={'ADD_PARTICIPANT','REMOVE_PARTICIPANT','MERGE_ALIASES'}
    assert np.isfinite(observed.action_features).all()
    model=GraphEditPolicy(width=32,layers=1)
    values=model(observation_tensors(observed))
    assert values.shape==(len(observed.edits),)
    values.sum().backward()
    assert model.node_input.weight.grad is not None


def test_action_arguments_reference_observed_nodes_only():
    graph=Graph('fixture',{'a':'Case','b':'Case'},[Event('e',0,'cancel',('a',))])
    state=RepairState(graph)
    observed=propose_actions(state,{'Case'},event_budget=None)
    assert all(e.target in graph.objects for e in observed.edits)
    assert all(e.source in (graph.objects if e.operation=='MERGE_ALIASES' else {'e'}) for e in observed.edits)
