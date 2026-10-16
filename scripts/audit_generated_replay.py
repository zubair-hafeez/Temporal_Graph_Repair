"""Check saved graph-level outcomes by replaying frozen inference independently."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_edit_policy import compare,thresholds_from_validation
from tgrepair.factory_plugin import learner_graph
from tgrepair.decoding import FeatureScorer
from tgrepair.view_decoding import select_fields,decode_with_view
from tgrepair.metrics import traceability
from tgrepair.data import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reports',nargs='+',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    protocol_path=ROOT/'docs/CONFIRMATION_PROTOCOL_V1.json'
    protocol=json.loads(protocol_path.read_text())
    for path,digest in protocol['implementation_hashes'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('Frozen code changed')
    results=[];torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        for name in args.reports:
            report_path=ROOT/'reports'/f'{name}.json';raw=report_path.read_bytes()
            report=json.loads(raw);config=report['config'];expected=report['results'][0]
            if report['protocol_hash']!=sha256(protocol_path) or config['kind']!='feature':raise ValueError('Unsupported replay')
            folder=ROOT/'data/generated'/config['generated']/expected['episode']/expected['variant']
            learner=json.loads((folder/'learner.json').read_text());reference=json.loads((folder/'reference.json').read_text())
            graph=learner_graph(learner);types=set(learner['schema']['target_types'])
            model=ROOT/'checkpoints'/(config['model']+'.joblib')
            if sha256(model)!=report['model_hash']:raise ValueError('Model changed')
            thresholds,_=thresholds_from_validation(config['calibration'] or config['model'],config['precision'])
            if thresholds!=report['thresholds']:raise ValueError('Thresholds changed')
            fields,view_profile=select_fields(graph,types,config['attribute_view'],config['stability_threshold'])
            start=time.monotonic()
            state,audit=decode_with_view(graph,types,FeatureScorer(model),thresholds,rounds=config['rounds'],mode=config['mode'],attribute_fields=fields)
            actual=compare(graph,state,reference,types)
            old_audit=json.loads((ROOT/'data/decoding'/config['output']/f"{expected['episode']}_{expected['variant']}.json").read_text())
            assert actual==expected['after'], 'Replayed application outcome differs'
            assert traceability(graph,reference,types)==expected['before'], 'Initial outcome differs'
            assert audit['edit_count']==expected['edit_count'] and audit['rounds']==old_audit['rounds'], 'Edit program differs'
            assert view_profile==expected['attribute_view'], 'Observation view differs'
            import hashlib
            results.append({'report':name,'report_snapshot_sha256':hashlib.sha256(raw).hexdigest(),
                            'episode':expected['episode'],'variant':expected['variant'],'edits':audit['edit_count'],
                            'seconds':time.monotonic()-start,'exact_outcome_and_edit_program_match':True})
    result={'status':'sampled_frozen_inference_replay_passed','protocol_hash':sha256(protocol_path),
            'script_hash':sha256(__file__),'replays':results,
            'limit':'Fixed first completed graph from each listed run; this is a reproducibility check, not additional statistical replication.'}
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
