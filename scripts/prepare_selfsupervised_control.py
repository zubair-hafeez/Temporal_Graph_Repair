"""A no-clean-label control: learn to undo extra faults on imperfect observations.

The baseline sees the same damaged fit graph as the annotation experiments, but
receives none of its original memberships or alias labels. Its training targets
are the pre-augmentation observations, which can themselves contain errors.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.corruption import mixed_faults
from tgrepair.data import Graph,load_graph,save_graph,sha256
from tgrepair.provenance import snapshot_sources
from prepare_source_edit_controls import build


def temporal_pseudo_tasks(observed,types):
    """Only an observation is accepted; there is no underlying-reference input."""
    boundary=observed.events[int(len(observed.events)*.7)].time
    tasks={}
    for split,seed in [('train',62011),('validation',62012)]:
        rows=[e for e in observed.events if (e.time<boundary)==(split=='train')]
        if not rows:raise ValueError('Empty pseudo-training time partition')
        ids={o for e in rows for o in e.participants}
        base=Graph(observed.name+'_'+split,{o:observed.objects[o] for o in sorted(ids)},rows)
        tasks[split]=mixed_faults(base,seed,types)
    return tasks,boundary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='selfsupervised_controls_v1')
    args=parser.parse_args();out=ROOT/'data/edit_learning'/args.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.monotonic();snapshot=snapshot_sources(ROOT)
    source=ROOT/'data/development/bpi2012_fit_attributes.json.gz'
    # Construction of the controlled damaged input is evaluation setup. Discard
    # its reference immediately; the baseline API below accepts only observed.
    observed,underlying_reference=mixed_faults(load_graph(source),0,{'Application'})
    del underlying_reference
    save_graph(observed,out/'imperfect_input.json.gz')
    tasks,boundary=temporal_pseudo_tasks(observed,{'Application'})
    report={'status':'development_no_clean_label_control_cache','source_snapshot':snapshot,
            'source_hash':sha256(source),'imperfect_input_hash':sha256(out/'imperfect_input.json.gz'),
            'time_boundary':boundary,'splits':{},
            'contract':'Train to reverse only added corruption. Existing erroneous observations are pseudo-reference labels, never secretly corrected.',
            'limits':['No original clean source labels enter model fitting or threshold selection.',
                      'Validation uses separately added faults in the later 30% time window, not real repair truth.',
                      'Some entity handles span both time windows; this is not an entity-disjoint calibration claim.',
                      'The BPI2012 amount observation is the declared denormalized track.']}
    for split,(graph,reference) in tasks.items():
        save_graph(graph,out/f'{split}_observed.json.gz')
        (out/f'{split}_pseudo_reference.json').write_text(json.dumps(reference,separators=(',',':'))+'\n')
        x,y,ops,counts=build(graph,reference,None,8192 if split=='train' else 2048)
        np.savez_compressed(out/f'{split}.npz',x=x,y=y,operation=ops)
        # build's legacy field counts sampled candidate-supervision events, not
        # verified histories in this no-clean-label experiment.
        counts['pseudo_reference_events']=counts.pop('verified_history_events')
        counts.pop('verified_entities',None)
        counts['actions']=len(y);counts['positives']=[int(y[ops==op].sum()) for op in range(3)]
        report['splits'][split]=counts
        print(json.dumps({'split':split,'counts':counts,'seconds':time.monotonic()-start}),flush=True)
    report['seconds']=time.monotonic()-start
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports'/f'{args.output}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
