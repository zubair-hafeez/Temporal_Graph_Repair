"""Reproducible illustrative success/failure selection, separate from inference."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=ROOT/args.task
    original=load_graph(folder/'observed.json.gz');reference=json.loads((folder/'reference.json').read_text())
    types=set(json.loads((folder/'task.json').read_text())['target_types'])
    result=json.loads((ROOT/'reports'/f'{args.run}.json').read_text())
    repaired=load_graph(ROOT/'data/decoding'/args.run/'repaired.json.gz')
    clusters=json.loads((ROOT/'data/decoding'/args.run/'clusters.json').read_text())
    handles=defaultdict(list);truth=defaultdict(set);before=defaultdict(set);after=defaultdict(set)
    event_lookup={e.uid:e for e in original.events}
    for obj,true in reference['identity'].items():
        if original.objects[obj] in types:handles[true].append(obj)
    for eid,objects in reference['event_participation'].items():
        for obj in objects:
            if original.objects[obj] in types:truth[obj].add(eid)
    for event in original.events:
        for obj in event.participants:
            if original.objects[obj] in types:before[obj].add(event.uid)
    for event in repaired.events:
        for obj in event.participants:
            if original.objects[obj] in types:after[clusters[obj]].add(event.uid)
    def f1(a,b):return 2*len(a&b)/(len(a)+len(b))
    rows=[]
    for true,wanted in truth.items():
        ids=handles[true]
        a=float(np.mean([f1(wanted,before[o]) for o in ids]))
        b=float(np.mean([f1(wanted,after[clusters[o]]) for o in ids]))
        exception=any(event_lookup[e].attributes.get('is_exception') in {1,True,'1'} or 'cancel' in event_lookup[e].kind.lower() for e in wanted)
        rows.append({'true':true,'type':original.objects[true],'before_f1':a,'after_f1':b,'gain':b-a,
                     'records':len(wanted),'has_exception_evidence':exception,'handles':sorted(ids)})
    if abs(float(np.mean([r['after_f1'] for r in rows]))-result['after']['history']['f1'])>1e-10:
        raise ValueError('Case-level reconstruction disagrees with reported metric')
    eligible=[r for r in rows if r['gain']>0 and r['has_exception_evidence']]
    selected=[]
    if eligible:
        median=float(np.median([r['gain'] for r in eligible]))
        selected.append(('exception_success_nearest_median_gain',min(eligible,key=lambda r:(abs(r['gain']-median),hashlib.sha256(r['true'].encode()).hexdigest()))))
    losses=[r for r in rows if r['gain']<0]
    if losses:selected.append(('largest_history_f1_loss',min(losses,key=lambda r:(r['gain'],hashlib.sha256(r['true'].encode()).hexdigest()))))
    cases=[]
    for rule,row in selected:
        true=row['true'];ids=row['handles']
        uid=min(ids,key=lambda o:hashlib.sha256(o.encode()).hexdigest())
        records=truth[true]|before[uid]|after[clusters[uid]]
        cases.append({'selection_rule':rule,**row,'queried_handle':uid,'events':[
            {'uid':e,'time':event_lookup[e].time,'activity':event_lookup[e].kind,
             'in_reference_history':e in truth[true],'retrieved_before':e in before[uid],
             'retrieved_after':e in after[clusters[uid]],'attributes':event_lookup[e].attributes}
            for e in sorted(records,key=lambda e:(event_lookup[e].time,e))]})
    report={'status':'posthoc_illustrative_case_audit','config':vars(args),'task_hash':sha256(folder/'task.json'),
            'result_hash':sha256(ROOT/'reports'/f'{args.run}.json'),'case_audit_script_hash':sha256(__file__),
            'population':{'entities':len(rows),'improved':sum(r['gain']>0 for r in rows),'worsened':len(losses),
                          'unchanged':sum(r['gain']==0 for r in rows)},'cases':cases,
            'limits':['Examples are selected from evaluated outcomes to illustrate behavior, not an independent efficacy test.',
                      'Entity scores average across aliases; the displayed record list uses one hash-selected observed handle.']}
    destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'population':report['population'],'cases':[{k:v for k,v in c.items() if k!='events'} for c in cases]}),flush=True)


if __name__=='__main__':main()
