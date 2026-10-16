"""Application-level development evaluation with observation-only decoding."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
from sklearn.metrics import precision_recall_curve
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.decoding import decode,NeuralScorer,FeatureScorer
from tgrepair.factory_plugin import learner_graph
from tgrepair.metrics import traceability,identity_pair_scores
from tgrepair.provenance import snapshot_sources
from tgrepair.data import sha256
from tgrepair.pipeline_faults import pipeline_faults
from tgrepair.view_decoding import select_fields,decode_with_view
from tgrepair.confirmation_contract import verify_model


def thresholds_from_validation(name,target_precision):
    cache=np.load(ROOT/'data/edit_learning'/f'{name}_validation_scores.npz')
    thresholds=[];details=[]
    for op in range(3):
        mask=cache['operation']==op
        y,p=cache['label'][mask],cache['probability'][mask]
        precision,recall,values=precision_recall_curve(y,p)
        feasible=np.flatnonzero((precision[:-1]>=target_precision)&(recall[:-1]*y.sum()>=5))
        if len(feasible):
            i=feasible[np.argmax(recall[feasible])]
            thresholds.append(float(values[i]));details.append({'precision':float(precision[i]),'recall':float(recall[i])})
        else:thresholds.append(1.1);details.append({'precision':None,'recall':0.})
    return thresholds,details


def compare(original,state,reference,types):
    identity=reference['identity'];before={e.uid:e for e in original.events};counts={'originally_correct_events':0,'preserved_correct_events':0,'damaged_events':0,'repaired_events':0}
    for e in state.graph.events:
        a=before[e.uid]
        if (a.time,a.kind,a.attributes)!=(e.time,e.kind,e.attributes):raise AssertionError('Protected event changed')
        expected={o for o in reference['event_participation'][e.uid] if original.objects[o] in types}
        old={identity[o] for o in a.participants if original.objects[o] in types}
        new={identity[o] for o in e.participants if original.objects[o] in types}
        if old==expected:
            counts['originally_correct_events']+=1;counts['preserved_correct_events']+=int(new==expected)
        else:
            counts['damaged_events']+=1;counts['repaired_events']+=int(new==expected)
    return {'history':traceability(state.graph,reference,types,state.clusters),
            'identity':identity_pair_scores(state.graph,reference,types,state.clusters),'events':counts,
            'protected_event_fields_preserved':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--kind',choices=['neural','feature'],required=True)
    p.add_argument('--generated',default='workflow_validation_v3');p.add_argument('--episodes',type=int,default=24)
    p.add_argument('--precision',type=float,default=.95);p.add_argument('--mode',choices=['joint','sequential','one_pass'],default='joint')
    p.add_argument('--calibration',help='Optional separately pinned validation-score cache; default uses model name')
    p.add_argument('--rounds',type=int,default=8);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--fault-family',choices=['original','pipeline'],default='original')
    p.add_argument('--attribute-view',choices=['all','none','stable'],default='all')
    p.add_argument('--stability-threshold',type=float,default=.5)
    p.add_argument('--confirmation-protocol')
    args=p.parse_args();destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    torch.set_num_threads(4);start=time.monotonic();snapshot=snapshot_sources(ROOT)
    thresholds,calibration=thresholds_from_validation(args.calibration or args.model,args.precision)
    model_path=ROOT/'checkpoints'/(args.model+('.pt' if args.kind=='neural' else '.joblib'))
    model_hash=sha256(model_path)
    protocol_hash=None
    if args.confirmation_protocol:
        protocol_path=ROOT/args.confirmation_protocol;protocol=json.loads(protocol_path.read_text())
        expected=protocol['generated_confirmation']
        if protocol['status']!='frozen_before_reserved_evaluation':raise ValueError('Protocol is not frozen')
        if args.generated!=expected['attribute_collection'] or args.episodes!=expected['count']:
            raise ValueError('Generated test collection differs from frozen protocol')
        if [args.rounds,16,32,args.precision]!=expected['decoder_contract']:raise ValueError('Generated decoder differs from protocol')
        verify_model(ROOT,protocol,{},args.model,model_path,args.calibration or args.model)
        for path,digest in protocol['implementation_hashes'].items():
            if sha256(ROOT/path)!=digest:raise ValueError('Frozen implementation changed: '+path)
        protocol_hash=sha256(protocol_path)
    scorer=NeuralScorer(model_path,args.device) if args.kind=='neural' else FeatureScorer(model_path)
    folder=ROOT/'data/generated'/args.generated;manifest=json.loads((folder/'dataset_manifest.json').read_text())
    episodes=random.Random(919).sample(manifest['episodes'],min(args.episodes,len(manifest['episodes'])))
    report={'status':'running_generated_confirmation' if protocol_hash else 'development_application_evaluation','config':vars(args),'source_snapshot':snapshot,
            'thresholds':thresholds,'model_hash':model_hash,'action_validation_calibration':calibration,'results':[]}
    report['calibration_cache_hash']=sha256(ROOT/'data/edit_learning'/f'{args.calibration or args.model}_validation_scores.npz')
    if protocol_hash:report['protocol_hash']=protocol_hash
    audit_folder=ROOT/'data/decoding'/args.output;audit_folder.mkdir(parents=True)
    with threadpool_limits(limits=4):
        for j,episode in enumerate(episodes):
            for variant in ['mixed','clean']:
                location=folder/episode['id']/variant
                learner=json.loads((location/'learner.json').read_text());reference=json.loads((location/'reference.json').read_text())
                graph=learner_graph(learner);types=set(learner['schema']['target_types'])
                if variant=='mixed' and args.fault_family=='pipeline':
                    clean=json.loads((folder/episode['id']/'clean'/'learner.json').read_text())
                    graph,reference=pipeline_faults(learner_graph(clean),int(episode['id'].split('-')[-1])+210001,types)
                baseline=traceability(graph,reference,types)
                fields,view_profile=select_fields(graph,types,args.attribute_view,args.stability_threshold)
                state,audit=decode_with_view(graph,types,scorer,thresholds,rounds=args.rounds,mode=args.mode,attribute_fields=fields)
                result={'episode':episode['id'],'group':episode['group'],'variant':variant,'projection':episode['projection'],
                        'before':baseline,'after':compare(graph,state,reference,types),'edit_count':audit['edit_count'],
                        'seconds':audit['seconds'],'stopped':audit['stopped'],'attribute_view':view_profile}
                (audit_folder/f"{episode['id']}_{variant}.json").write_text(json.dumps(audit)+'\n')
                report['results'].append(result);report['seconds']=time.monotonic()-start
                destination.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'completed':j+1,'seconds':time.monotonic()-start,
                              'mixed_before':report['results'][-2]['before']['f1'],'mixed_after':report['results'][-2]['after']['history']['f1'],
                              'clean_after':report['results'][-1]['after']['history']['f1']}),flush=True)
    summary={}
    for variant in ['mixed','clean']:
        rows=[r for r in report['results'] if r['variant']==variant]
        summary[variant]={'mean_before_f1':float(np.mean([r['before']['f1'] for r in rows])),
                          'mean_after_f1':float(np.mean([r['after']['history']['f1'] for r in rows])),
                          'mean_seconds':float(np.mean([r['seconds'] for r in rows])),
                          'edits':sum(r['edit_count'] for r in rows)}
    if sha256(model_path)!=model_hash:raise RuntimeError('Model changed during evaluation')
    report['summary']=summary
    if protocol_hash:report['status']='completed_generated_confirmation'
    destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
