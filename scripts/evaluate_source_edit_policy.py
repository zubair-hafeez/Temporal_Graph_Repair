"""Zero-shot synthetic edit learning on the already explored real-source tune log."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,save_graph,sha256
from tgrepair.decoding import FeatureScorer,NeuralScorer,decode
from tgrepair.metrics import traceability
from tgrepair.provenance import snapshot_sources
from tgrepair.view_decoding import select_fields,decode_with_view
from evaluate_edit_policy import compare,thresholds_from_validation


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--kind',choices=['feature','neural'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--participant-budget',type=int,default=128);p.add_argument('--pair-budget',type=int,default=64)
    p.add_argument('--precision',type=float,default=.95)
    p.add_argument('--calibration',help='Optional separately pinned validation-score cache')
    p.add_argument('--regime',choices=['synthetic_only','source_only','mixed','self_supervised','self_supervised_plus_generated'],default='synthetic_only')
    p.add_argument('--calibration-domain',choices=['generated','source_development','pseudo_development'],default='generated')
    p.add_argument('--verified-histories',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--attribute-view',choices=['all','none','stable'],default='all')
    p.add_argument('--stability-threshold',type=float,default=.5)
    args=p.parse_args();destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists() and not args.resume:raise FileExistsError(destination)
    start=time.monotonic();snapshot=snapshot_sources(ROOT);torch.set_num_threads(4)
    directory=ROOT/'data/sequential_development_v1';graph=load_graph(directory/'observed.json.gz')
    reference=json.loads((directory/'reference.json').read_text());types={'Application'}
    fields,view_profile=select_fields(graph,types,args.attribute_view,args.stability_threshold)
    model_path=ROOT/'checkpoints'/(args.model+('.pt' if args.kind=='neural' else '.joblib'));model_hash=sha256(model_path)
    scorer=FeatureScorer(model_path) if args.kind=='feature' else NeuralScorer(model_path)
    thresholds,calibration=thresholds_from_validation(args.calibration or args.model,args.precision)
    out=ROOT/'data/decoding'/args.output;out.mkdir(parents=True,exist_ok=args.resume)
    checkpoint_path=out/'checkpoint.json'
    pins={'source_hash':sha256(directory/'observed.json.gz'),'reference_hash':sha256(directory/'reference.json'),'model_hash':model_hash,
          'thresholds':thresholds,'rounds':args.rounds,'participant_budget':args.participant_budget,'pair_budget':args.pair_budget,
          'implementation':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'tgrepair'/f'{name}.py' for name in
                            ['edits','decoding','matching','entity_features','action_features','neural_policy','data']]}}
    if args.attribute_view!='all':
        pins['attribute_view']=view_profile
        pins['implementation'].update({f'tgrepair/{name}.py':sha256(ROOT/'tgrepair'/f'{name}.py') for name in ['attribute_views','view_decoding']})
    resume_history=None;prior_elapsed=0.
    if args.resume:
        saved=json.loads(checkpoint_path.read_text())
        if saved['pins']!=pins:raise ValueError('Resume requires identical model, inputs, thresholds, budgets and implementation')
        resume_history=saved['history']
        prior_elapsed=saved['elapsed_seconds']
    def checkpoint(history):
        temporary=out/'checkpoint.tmp'
        temporary.write_text(json.dumps({'pins':pins,'history':history,'elapsed_seconds':prior_elapsed+time.monotonic()-start})+'\n')
        temporary.replace(checkpoint_path)
    report={'status':'running_development_source_evaluation','config':vars(args),'source_snapshot':snapshot,
            'model_hash':model_hash,'source_hash':sha256(directory/'observed.json.gz'),
            'calibration_cache_hash':sha256(ROOT/'data/edit_learning'/f'{args.calibration or args.model}_validation_scores.npz'),
            'thresholds':thresholds,'calibration':calibration,'before':traceability(graph,reference,types),
            'limits':['Previously inspected BPI2012 tune, not a final target.',
                      ({'generated':'Thresholds use generated validation only; no source labels enter decoding.',
                        'source_development':'Thresholds use a source-development sample from this evaluation log; this is a development result, not an independent test or total annotation-budget claim.',
                        'pseudo_development':'Thresholds use added faults on already imperfect fit observations. No true source reference labels enter fitting or calibration.'}[args.calibration_domain]),
                      'This policy receives target entity types but no singleton-participation constraint.']}
    report['resumed_rounds']=len(resume_history or [])
    report['attribute_view']=view_profile
    destination.write_text(json.dumps(report,indent=2)+'\n')
    with threadpool_limits(limits=4):
        state,audit=decode_with_view(graph,types,scorer,thresholds,rounds=args.rounds,participant_budget=args.participant_budget,
                           pair_budget=args.pair_budget,progress=lambda r:print(json.dumps(r),flush=True),
                           resume_history=resume_history,checkpoint=checkpoint,attribute_fields=fields)
    report['after']=compare(graph,state,reference,types);report['seconds']=prior_elapsed+time.monotonic()-start
    report['status']='completed_development_source_evaluation';report['edit_count']=audit['edit_count']
    save_graph(state.graph,out/'repaired.json.gz');(out/'clusters.json').write_text(json.dumps(state.clusters)+'\n')
    (out/'audit.json').write_text(json.dumps(audit)+'\n')
    if sha256(model_path)!=model_hash:raise RuntimeError('Model changed during evaluation')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'before':report['before'],'after':report['after'],'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
