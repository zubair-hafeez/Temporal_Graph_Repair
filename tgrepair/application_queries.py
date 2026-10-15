"""Evaluator-only operational queries over reconstructed entity histories."""
from collections import defaultdict


def audit_queries(graph,reference,target_types,clusters=None,exception_event_ids=None):
    clusters=clusters or {o:o for o in graph.objects}
    identity=reference['identity'];handles=defaultdict(list);expected=defaultdict(set);predicted=defaultdict(set)
    times={event.uid:event.time for event in graph.events}
    for obj,true in identity.items():
        if graph.objects[obj] in target_types:handles[true].append(obj)
    for eid,objects in reference['event_participation'].items():
        for obj in objects:
            if graph.objects[obj] in target_types:expected[obj].add(eid)
    for event in graph.events:
        for obj in event.participants:
            if graph.objects[obj] in target_types:predicted[clusters[obj]].add(event.uid)
    # Query definition is explicit and auditable, not a claim that every cancel
    # activity means the whole business transaction was canceled.
    cancellation={e.uid for e in graph.events if 'cancel' in e.kind.lower()}
    exceptional=set(exception_event_ids or [])
    counts={'entities':0,'empty_history':0.,'first_record_correct':0.,'last_record_correct':0.}
    flags={name:{'true_positive':0.,'false_positive':0.,'false_negative':0.,'true_negative':0.}
           for name in ['cancellation_evidence','exception_evidence']}
    for true,ids in handles.items():
        truth=expected[true]
        if not truth:continue
        counts['entities']+=1
        start=min(times[e] for e in truth);finish=max(times[e] for e in truth)
        for obj in ids:
            weight=1/len(ids);history=predicted[clusters[obj]]
            counts['empty_history']+=weight*int(not history)
            counts['first_record_correct']+=weight*int(bool(history) and min(times[e] for e in history)==start)
            counts['last_record_correct']+=weight*int(bool(history) and max(times[e] for e in history)==finish)
            for name,relevant in [('cancellation_evidence',cancellation),('exception_evidence',exceptional)]:
                wanted=bool(truth&relevant);answer=bool(history&relevant)
                key=('true_positive' if answer else 'false_negative') if wanted else ('false_positive' if answer else 'true_negative')
                flags[name][key]+=weight
    n=max(1,counts['entities'])
    result={'entities':counts['entities'],'empty_history_fraction':counts['empty_history']/n,
            'first_timestamp_correct_fraction':counts['first_record_correct']/n,'last_timestamp_correct_fraction':counts['last_record_correct']/n,
            'queries':{}}
    for name,c in flags.items():
        tp,fp,fn=c['true_positive'],c['false_positive'],c['false_negative']
        result['queries'][name]={**c,'precision':tp/(tp+fp) if tp+fp else None,
                                 'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    return result
