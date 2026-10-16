"""Target-local denoising control that never opens a clean task reference."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,save_graph,sha256
from tgrepair.provenance import snapshot_sources
from prepare_selfsupervised_control import temporal_pseudo_tasks
from prepare_source_edit_controls import build


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    task_dir=ROOT/args.task
    task=json.loads((task_dir/'task.json').read_text())
    observation_path=task_dir/'observed.json.gz'
    if sha256(observation_path)!=task['observation_sha256']:
        raise ValueError('Observation hash mismatch')
    observed=load_graph(observation_path)
    types=set(task['target_types'])
    out=ROOT/'data/edit_learning'/args.output
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    tasks,boundary=temporal_pseudo_tasks(observed,types)
    report={'status':'target_local_selfsupervision_cache','config':vars(args),
            'source_snapshot':snapshot_sources(ROOT),'observation_hash':sha256(observation_path),
            'target_types':sorted(types),'time_boundary':boundary,'splits':{},
            'contract':'No reference.json is read. The imperfect task observation is the pseudo target for independently added faults.',
            'limits':['Transductive target adaptation; the evaluated observation is available without clean labels.',
                      'Time partitions may share entity handles. Existing observation errors remain noisy pseudo labels.']}
    for split,(graph,reference) in tasks.items():
        save_graph(graph,out/f'{split}_observed.json.gz')
        (out/f'{split}_pseudo_reference.json').write_text(json.dumps(reference,separators=(',',':'))+'\n')
        x,y,ops,counts=build(graph,reference,None,8192 if split=='train' else 2048,types)
        np.savez_compressed(out/f'{split}.npz',x=x,y=y,operation=ops)
        counts['pseudo_reference_events']=counts.pop('verified_history_events')
        counts.pop('verified_entities',None)
        counts['actions']=len(y)
        counts['positives']=[int(y[ops==op].sum()) for op in range(3)]
        report['splits'][split]=counts
        print(json.dumps({'split':split,'counts':counts,'seconds':time.monotonic()-start}),flush=True)
    report['seconds']=time.monotonic()-start
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports'/f'{args.output}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
