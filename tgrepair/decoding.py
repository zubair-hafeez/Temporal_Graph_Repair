"""Iterative graph-edit decoding. No reference data enter inference."""
from collections import Counter
from dataclasses import asdict
import time

import numpy as np

from .action_features import propose_actions,OPERATIONS
from .edits import RepairState,Edit


def decode(graph,target_types,scorer,thresholds,rounds=8,mode='joint',participant_budget=16,pair_budget=32,progress=None,
           resume_history=None,checkpoint=None):
    if mode not in {'joint','sequential','one_pass'}:raise ValueError(mode)
    state=RepairState(graph);history=[];applied=set();applied_keys=set();start=time.monotonic()
    if resume_history:
        for record in resume_history:
            edits=[Edit(**item['edit']) for item in record['edits']]
            state.apply_batch(edits)
            applied.update(edits)
            applied_keys.update((edit.operation,edit.source,edit.target) for edit in edits)
            history.append(record)
    already_stopped=bool(history and not history[-1]['edits'] and (mode!='sequential' or history[-1]['stage']=='identity'))
    for iteration in range(len(history),rounds) if not already_stopped else []:
        observation=propose_actions(state,target_types,seed=0,event_budget=None,
                                    participant_budget=participant_budget,pair_budget=pair_budget)
        scores=scorer(observation)
        if len(scores)!=len(observation.edits):raise ValueError('Score shape mismatch')
        selected=[];used_events=set();touched=set();merged=set()
        stage=('participation' if iteration<rounds//2 else 'identity') if mode=='sequential' else 'all'
        for i in np.argsort(-scores,kind='stable'):
            edit=observation.edits[i];op=observation.operations[i]
            if scores[i]<thresholds[op] or edit in applied:continue
            if stage=='participation' and op==2 or stage=='identity' and op!=2:continue
            # A batch consists of commuting edits: no merged entity is also edited,
            # and at most one participation change per event/type is made per pass.
            if op==2:
                if {edit.source,edit.target}&touched:continue
                touched.update([edit.source,edit.target]);merged.update([edit.source,edit.target])
            else:
                event_key=(edit.source,state.graph.objects[edit.target])
                if event_key in used_events or edit.target in merged:continue
                inverse=('REMOVE_PARTICIPANT' if op==0 else 'ADD_PARTICIPANT',edit.source,edit.target)
                if inverse in applied_keys:continue
                used_events.add(event_key);touched.add(edit.target)
            selected.append((edit,float(scores[i])))
        state.apply_batch([edit for edit,score in selected])
        for edit,score in selected:
            applied.add(edit);applied_keys.add((edit.operation,edit.source,edit.target))
        history.append({'round':iteration+1,'stage':stage,'candidates':len(scores),
                        'counts':dict(Counter(edit.operation for edit,_ in selected)),
                        'edits':[{'edit':asdict(edit),'score':score} for edit,score in selected]})
        if progress is not None:
            progress({'round':iteration+1,'candidates':len(scores),'accepted':len(selected),
                      'counts':history[-1]['counts'],'seconds':time.monotonic()-start})
        if checkpoint is not None:
            checkpoint(history)
        if mode=='one_pass':break
        if not selected and (mode!='sequential' or stage=='identity'):break
    return state,{'rounds':history,'seconds':time.monotonic()-start,'stopped':not history[-1]['edits'] if history else True,
                  'edit_count':len(applied),'thresholds':list(thresholds),'mode':mode,
                  'participant_budget':participant_budget,'pair_budget':pair_budget}


class FeatureScorer:
    def __init__(self,path):
        import joblib
        self.models=joblib.load(path)

    def __call__(self,observation):
        result=np.zeros(len(observation.edits))
        for op,model in enumerate(self.models):
            mask=observation.operations==op
            if mask.any():result[mask]=model.predict_proba(observation.action_features[mask])[:,1]
        return result


class NeuralScorer:
    def __init__(self,path,device='cpu'):
        import torch
        from .neural_policy import GraphEditPolicy
        self.torch=torch;self.device=device
        checkpoint=torch.load(path,map_location=device,weights_only=True)
        self.model=GraphEditPolicy(**checkpoint['architecture']).to(device)
        self.model.load_state_dict(checkpoint['model']);self.model.eval()
        self.mean=checkpoint['mean'].to(device);self.scale=checkpoint['scale'].to(device)

    def __call__(self,observation):
        from .neural_policy import observation_tensors
        data=observation_tensors(observation,self.device)
        data['action_features']=(data['action_features']-self.mean)/self.scale
        with self.torch.no_grad():return self.torch.sigmoid(self.model(data)).cpu().numpy()
