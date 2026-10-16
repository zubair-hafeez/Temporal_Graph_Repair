"""Pinned, resumable repair evaluation on one independently generated task."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,save_graph,sha256
from tgrepair.decoding import FeatureScorer,NeuralScorer,decode
from tgrepair.metrics import traceability
from tgrepair.application_queries import audit_queries
from tgrepair.provenance import snapshot_sources
from tgrepair.view_decoding import select_fields,decode_with_view
from tgrepair.confirmation_contract import verify_model
from evaluate_edit_policy import compare,thresholds_from_validation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    parser.add_argument('--model',default='edit_feature_v2')
    parser.add_argument('--kind',choices=['feature','neural'],default='feature')
    parser.add_argument('--output',required=True)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--participant-budget',type=int,default=128)
    parser.add_argument('--pair-budget',type=int,default=64)
    parser.add_argument('--precision',type=float,default=.95)
    parser.add_argument('--calibration',help='Optional separately pinned validation-score cache')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--attribute-view',choices=['all','none','stable'],default='all')
    parser.add_argument('--stability-threshold',type=float,default=.5)
    parser.add_argument('--protocol',help='Required frozen protocol for reserved confirmation tasks')
    parser.add_argument('--mode',choices=['joint','sequential','one_pass'],default='joint')
    args=parser.parse_args()
    destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists() and not args.resume:raise FileExistsError(destination)
    directory=ROOT/args.task
    task=json.loads((directory/'task.json').read_text())
    confirmation=bool(args.protocol)
    protocol=None
    if confirmation:
        protocol_path=ROOT/args.protocol;protocol=json.loads(protocol_path.read_text())
        if protocol['status']!='frozen_before_reserved_evaluation' or task.get('protocol_hash')!=sha256(protocol_path):
            raise ValueError('Reserved evaluation requires the exact frozen task protocol')
        if task['domain'] not in protocol['sources']:raise ValueError('Unregistered reserved source')
        if [args.rounds,args.participant_budget,args.pair_budget,args.precision]!=protocol['decoder_contract']:
            raise ValueError('Decoder options differ from frozen protocol')
        for path,digest in protocol['implementation_hashes'].items():
            if sha256(ROOT/path)!=digest:raise ValueError('Frozen evaluation implementation changed: '+path)
    elif task['domain'] not in {'batteryvault','claimstream'}:
        raise ValueError('Reserved scenarios require --protocol')
    for field,filename in [('observation_sha256','observed.json.gz'),('reference_sha256','reference.json')]:
        if task[field]!=sha256(directory/filename):raise ValueError('Task artifact hash mismatch')
    start=time.monotonic();snapshot=snapshot_sources(ROOT);torch.set_num_threads(4)
    graph=load_graph(directory/'observed.json.gz');reference=json.loads((directory/'reference.json').read_text())
    types=set(task['target_types'])
    fields,view_profile=select_fields(graph,types,args.attribute_view,args.stability_threshold)
    model_path=ROOT/'checkpoints'/(args.model+('.pt' if args.kind=='neural' else '.joblib'))
    if confirmation:
        verify_model(ROOT,protocol,task,args.model,model_path,args.calibration or args.model)
    scorer=FeatureScorer(model_path) if args.kind=='feature' else NeuralScorer(model_path)
    thresholds,calibration=thresholds_from_validation(args.calibration or args.model,args.precision)
    out=ROOT/'data/decoding'/args.output;out.mkdir(parents=True,exist_ok=args.resume)
    checkpoint_path=out/'checkpoint.json'
    pins={'task_hash':sha256(directory/'task.json'),'model_hash':sha256(model_path),
          'thresholds':thresholds,'rounds':args.rounds,'participant_budget':args.participant_budget,
          'pair_budget':args.pair_budget,'implementation':{
              str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'tgrepair'/f'{name}.py' for name in
                  ['edits','decoding','matching','entity_features','action_features','neural_policy','data']]}}
    if confirmation:pins['protocol_hash']=sha256(protocol_path)
    if args.mode!='joint':pins['mode']=args.mode
    if args.attribute_view!='all':
        pins['attribute_view']=view_profile
        pins['implementation'].update({f'tgrepair/{name}.py':sha256(ROOT/'tgrepair'/f'{name}.py') for name in ['attribute_views','view_decoding']})
    resume_history=None;prior_elapsed=0.
    if args.resume:
        saved=json.loads(checkpoint_path.read_text())
        if saved['pins']!=pins:raise ValueError('Resume requires identical inputs, model, options and implementation')
        resume_history=saved['history'];prior_elapsed=saved['elapsed_seconds']
    def checkpoint(history):
        temporary=out/'checkpoint.tmp'
        temporary.write_text(json.dumps({'pins':pins,'history':history,'elapsed_seconds':prior_elapsed+time.monotonic()-start})+'\n')
        temporary.replace(checkpoint_path)
    # Evaluation labels must survive a structural-only learner view. Resolve its
    # pinned parent observations outside the decoder rather than treating hidden
    # exception flags as negative labels.
    exception_graph=graph;exception_source=directory/'observed.json.gz'
    if task.get('parent_task_hash'):
        parent=ROOT/'data/external_development_v1'/task['domain']/task['variant']
        if sha256(parent/'task.json')!=task['parent_task_hash']:raise ValueError('Query-label parent hash mismatch')
        exception_graph=load_graph(parent/'observed.json.gz');exception_source=parent/'observed.json.gz'
    exceptions=set(reference.get('exception_events',[])) if 'exception_events' in reference else {
        e.uid for e in exception_graph.events if e.attributes.get('is_exception') in {1,True,'1'}}
    report={'status':'running_confirmation_evaluation' if confirmation else 'running_external_development','config':vars(args),'task':task,
            'source_snapshot':snapshot,'pins':pins,'thresholds':thresholds,'calibration':calibration,
            'calibration_cache_hash':sha256(ROOT/'data/edit_learning'/f'{args.calibration or args.model}_validation_scores.npz'),
            'before':traceability(graph,reference,types),
            'queries_before':audit_queries(graph,reference,types,exception_event_ids=exceptions),
            'exception_query_source_hash':sha256(exception_source),
            'attribute_view':view_profile,
            'resumed_rounds':len(resume_history or []),
            'limits':['Controlled recording faults on public-source references, not independently labelled natural errors.',
                      'Calibration provenance is recorded separately; target evaluation labels never enter decoding.',
                      'Queries concern the available window; lifetime histories may be censored.']}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    with threadpool_limits(limits=4):
        state,audit=decode_with_view(graph,types,scorer,thresholds,rounds=args.rounds,
                           participant_budget=args.participant_budget,pair_budget=args.pair_budget,
                           progress=lambda r:print(json.dumps(r),flush=True),
                           resume_history=resume_history,checkpoint=checkpoint,attribute_fields=fields,mode=args.mode)
    report['after']=compare(graph,state,reference,types)
    report['queries_after']=audit_queries(state.graph,reference,types,state.clusters,exceptions)
    report['seconds']=prior_elapsed+time.monotonic()-start
    report['edit_count']=audit['edit_count'];report['status']='completed_confirmation_evaluation' if confirmation else 'completed_external_development'
    save_graph(state.graph,out/'repaired.json.gz')
    (out/'clusters.json').write_text(json.dumps(state.clusters)+'\n')
    (out/'audit.json').write_text(json.dumps(audit)+'\n')
    if sha256(model_path)!=pins['model_hash']:raise RuntimeError('Model changed during evaluation')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'before':report['before'],'after':report['after'],'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
