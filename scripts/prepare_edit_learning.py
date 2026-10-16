"""Cache observation-only graph/action tensors and separate training labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.action_features import propose_actions,OPERATIONS
from tgrepair.edits import RepairState,teacher_edits,beneficial
from tgrepair.factory_plugin import learner_graph
from tgrepair.neural_policy import observation_tensors
from tgrepair.provenance import snapshot_sources
from tgrepair.view_decoding import select_fields,propose_in_view


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--generated',required=True)
    p.add_argument('--episodes',type=int,default=512)
    p.add_argument('--output',required=True)
    p.add_argument('--partial',action='store_true')
    p.add_argument('--event-budget',type=int,default=48)
    p.add_argument('--participant-budget',type=int,default=16)
    p.add_argument('--pair-budget',type=int,default=32)
    p.add_argument('--attribute-view',choices=['all','none','stable'],default='all')
    p.add_argument('--stability-threshold',type=float,default=.5)
    args=p.parse_args()
    folder=ROOT/'data/edit_learning'/args.output
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    snapshot=snapshot_sources(ROOT);start=time.monotonic();counts=Counter();records=[]
    source=ROOT/'data/generated'/args.generated
    manifest=json.loads((source/'dataset_manifest.json').read_text())
    rng=random.Random(901)
    for i,episode in enumerate(manifest['episodes'][:args.episodes]):
        variants=['mixed','clean'] if i%4==0 else ['mixed']
        for variant in variants:
            location=source/episode['id']/variant
            learner=json.loads((location/'learner.json').read_text())
            reference=json.loads((location/'reference.json').read_text())
            graph=learner_graph(learner);types=set(learner['schema']['target_types'])
            fields,view_profile=select_fields(graph,types,args.attribute_view,args.stability_threshold)
            state=RepairState(graph)
            stages=['raw','partial'] if args.partial and variant=='mixed' and i%2==0 else ['raw']
            for stage in stages:
                if stage=='partial':
                    program=teacher_edits(state,reference,types)
                    for edit in program[:int(len(program)*rng.uniform(.2,.8))]:state.apply(edit)
                # Candidate construction has no access to the teacher or reference.
                observation=propose_in_view(state,types,attribute_fields=fields,seed=i,event_budget=args.event_budget,
                                            participant_budget=args.participant_budget,pair_budget=args.pair_budget)
                labels=np.asarray([beneficial(edit,state,reference) for edit in observation.edits],dtype=np.float32)
                data=observation_tensors(observation)
                filename=f'{len(records):06d}.pt'
                torch.save({'observation':data,'labels':torch.from_numpy(labels)},folder/filename)
                records.append({'file':filename,'episode':episode['id'],'group':episode['group'],
                                'variant':variant,'stage':stage,'actions':len(labels),'positive':int(labels.sum()),
                                'attribute_view':view_profile})
                for op,name in enumerate(OPERATIONS):
                    mask=observation.operations==op
                    counts[name+'_actions']+=int(mask.sum());counts[name+'_positive']+=int(labels[mask].sum())
                counts['states']+=1;counts['nodes']+=len(observation.node_features)
        if i%16==0:print(json.dumps({'episodes':i+1,'states':counts['states'],'seconds':time.monotonic()-start}),flush=True)
    report={'status':'development_edit_learning_cache','config':vars(args),'counts':dict(counts),
            'source_snapshot':snapshot,'source_data':args.generated,'records':records,'seconds':time.monotonic()-start}
    (folder/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports'/f'{args.output}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
