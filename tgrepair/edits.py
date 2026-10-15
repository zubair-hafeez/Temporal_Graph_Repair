"""Typed graph edits with explicit state, audit trail, and evaluator-only teacher."""
from dataclasses import dataclass, replace
from collections import defaultdict

from .data import Graph


@dataclass(frozen=True)
class Edit:
    operation: str
    source: str
    target: str


class RepairState:
    def __init__(self, graph):
        self.graph=Graph(graph.name,dict(graph.objects),list(graph.events))
        self.clusters={o:o for o in graph.objects}
        self.log=[]

    def active_graph(self):
        return Graph(self.graph.name,{o:t for o,t in self.graph.objects.items() if self.clusters[o]==o},list(self.graph.events))

    def apply(self,edit):
        if edit.operation not in {'ADD_PARTICIPANT','REMOVE_PARTICIPANT','MERGE_ALIASES'}:
            raise ValueError('Unsupported graph edit')
        prior_events={}
        prior_clusters={}
        if edit.operation=='MERGE_ALIASES':
            a,b=edit.source,edit.target
            if a not in self.clusters or b not in self.clusters:
                raise ValueError('Unknown identity handle')
            a,b=self.clusters[a],self.clusters[b]
            if a==b or self.graph.objects[a]!=self.graph.objects[b]:
                raise ValueError('Identity merge must join distinct same-type components')
            keep,remove=sorted((a,b))
            for o,current in self.clusters.items():
                if current==remove:
                    prior_clusters[o]=current
                    self.clusters[o]=keep
            events=[]
            for e in self.graph.events:
                if remove in e.participants:
                    prior_events[e.uid]=e.participants
                    e=replace(e,participants=tuple(sorted({keep if o==remove else o for o in e.participants})))
                events.append(e)
            self.graph.events=events
        else:
            if edit.target not in self.clusters or self.clusters[edit.target]!=edit.target:
                raise ValueError('Participation edits must reference an active entity handle')
            found=False;events=[]
            for e in self.graph.events:
                if e.uid==edit.source:
                    found=True;participants=set(e.participants)
                    if edit.operation=='ADD_PARTICIPANT':
                        if edit.target in participants:raise ValueError('Redundant participant addition')
                        participants.add(edit.target)
                    else:
                        if edit.target not in participants:raise ValueError('Cannot remove absent participation')
                        participants.remove(edit.target)
                    prior_events[e.uid]=e.participants
                    e=replace(e,participants=tuple(sorted(participants)))
                events.append(e)
            if not found:raise ValueError('Unknown event handle')
            self.graph.events=events
        self.log.append({'edit':edit,'prior_events':prior_events,'prior_clusters':prior_clusters})
        return self

    def undo(self):
        if not self.log:raise ValueError('No edit to undo')
        record=self.log.pop()
        self.clusters.update(record['prior_clusters'])
        self.graph.events=[replace(e,participants=record['prior_events'][e.uid]) if e.uid in record['prior_events'] else e for e in self.graph.events]

    def apply_batch(self, edits):
        """Apply commuting edits in one graph traversal; undo restores the batch."""
        edits=list(edits)
        if not edits:return self
        events={e.uid:e for e in self.graph.events};changes=defaultdict(list)
        merged=set();remap={};participants=set()
        for edit in edits:
            if edit.operation=='MERGE_ALIASES':
                a,b=edit.source,edit.target
                if a not in self.clusters or b not in self.clusters or self.clusters[a]!=a or self.clusters[b]!=b:
                    raise ValueError('Batch merge requires active handles')
                if a==b or self.graph.objects[a]!=self.graph.objects[b] or {a,b}&merged:
                    raise ValueError('Batch merges must be disjoint and same-type')
                keep,remove=sorted((a,b));merged.update([a,b]);remap[remove]=keep
            elif edit.operation in {'ADD_PARTICIPANT','REMOVE_PARTICIPANT'}:
                if edit.source not in events or edit.target not in self.clusters or self.clusters[edit.target]!=edit.target:
                    raise ValueError('Invalid batch participation endpoint')
                present=edit.target in events[edit.source].participants
                if present==(edit.operation=='ADD_PARTICIPANT'):
                    raise ValueError('Redundant batch participation edit')
                if edit in changes[edit.source]:raise ValueError('Duplicate batch edit')
                changes[edit.source].append(edit);participants.add(edit.target)
            else:raise ValueError('Unsupported batch edit')
        if merged&participants:raise ValueError('Batch merge and participation actions do not commute')
        prior_events={};prior_clusters={};result=[]
        for event in self.graph.events:
            values={remap.get(o,o) for o in event.participants}
            for edit in changes.get(event.uid,[]):
                if edit.operation=='ADD_PARTICIPANT':values.add(edit.target)
                else:values.remove(edit.target)
            if values!=set(event.participants):
                prior_events[event.uid]=event.participants;event=replace(event,participants=tuple(sorted(values)))
            result.append(event)
        for o,c in self.clusters.items():
            if c in remap:prior_clusters[o]=c;self.clusters[o]=remap[c]
        self.graph.events=result
        self.log.append({'edits':edits,'prior_events':prior_events,'prior_clusters':prior_clusters})
        return self


def teacher_edits(state,reference,target_types):
    """Evaluator/training-only canonical program; never called at inference."""
    groups=defaultdict(set)
    for o,true in reference['identity'].items():
        if state.graph.objects[o] in target_types:
            groups[true].add(state.clusters[o])
    edits=[]
    working=RepairState(state.graph)
    working.clusters=dict(state.clusters)
    for ids in groups.values():
        ids=sorted(ids)
        for other in ids[1:]:
            edit=Edit('MERGE_ALIASES',ids[0],other)
            working.apply(edit);edits.append(edit)
    expected={eid:{working.clusters[o] for o in objects} for eid,objects in reference['event_participation'].items()}
    for e in working.graph.events:
        observed=set(e.participants)
        for o in sorted(observed-expected[e.uid]):
            if working.graph.objects[o] in target_types:edits.append(Edit('REMOVE_PARTICIPANT',e.uid,o))
        for o in sorted(expected[e.uid]-observed):
            if working.graph.objects[o] in target_types:edits.append(Edit('ADD_PARTICIPANT',e.uid,o))
    return edits


def beneficial(edit,state,reference):
    """Training label for a candidate action; reference is evaluator-only."""
    identity=reference['identity']
    if edit.operation=='MERGE_ALIASES':
        return identity[edit.source]==identity[edit.target]
    expected=set(reference['event_participation'][edit.source])
    if edit.operation=='REMOVE_PARTICIPANT':
        return identity[edit.target] not in expected
    event=next(e for e in state.graph.events if e.uid==edit.source)
    existing={identity[o] for o in event.participants}
    return identity[edit.target] in expected-existing
