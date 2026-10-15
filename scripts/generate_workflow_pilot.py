"""Generate a bounded, auditable training pilot through factory task plugins."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.factory_plugin import registry, learner_graph
from tgrepair.data import sha256
from tgrepair.provenance import snapshot_sources
from temporal_factory.core import export_task, fingerprint, verify_export


TRAIN_COMPOSITIONS = [
    ['queues','handoff'],
    ['queues','handoff','retry'],
    ['queues','handoff','cancel_return'],
    ['queues','branch_join','retry'],
    ['queues','batch_split','handoff'],
    ['queues','batch_split','cancel_return'],
    ['queues','branch_join','cancel_return'],
    ['queues','branch_join','batch_split'],
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--jobs',type=int,default=12)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--output',default='workflow_pilot_v1')
    args=p.parse_args()
    if not 1<=args.count<=2000: raise ValueError('Pilot generation count must be in [1,2000]')
    output=ROOT/'data/generated'/args.output
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    source_snapshot=snapshot_sources(ROOT)
    modules=registry();rng=random.Random(args.seed)
    counts=Counter();episodes=[];start=time.monotonic()
    for i in range(args.count):
        mechanisms=TRAIN_COMPOSITIONS[i%len(TRAIN_COMPOSITIONS)]
        recipe={'seed':args.seed+i,'family':['service','fulfillment','assembly'][i%3],
                'jobs':args.jobs,'max_children':rng.choice([1,2,3,4]),'resources':rng.choice([2,4,8]),
                'arrival_interval':rng.choice([.5,2.,8.]),'service_scale':rng.choice([.5,1.,3.]),
                'attribute_palette':rng.choice([2,4,8,16]),'mechanisms':mechanisms}
        execution=modules.create_world('workflow',recipe).run()
        roles=execution['schema']['entity_roles']
        flat = i % 2 == 0
        visible_types = [roles['child'],roles['resource']] if flat else execution['schema']['entity_types']
        target_types = [roles['child']] if flat else [roles['parent'],roles['child']]
        episode_id=f'episode-{i:06d}'
        group=fingerprint({'recipe':execution['provenance']['recipe_hash'],'source':execution['provenance']['source_hash']})
        variants=[]
        for variant in ['mixed','clean']:
            config={'target_types':target_types,'observation_seed':args.seed+100000+i,
                    'visible_types':visible_types,
                    'missing_rate':.1 if variant=='mixed' else 0.,'wrong_rate':.1 if variant=='mixed' else 0.,
                    'fragment_rate':.25 if variant=='mixed' else 0.,'gap_mode':'contiguous'}
            artifact=modules.build_task('temporal_graph_repair',execution,config)
            location=output/episode_id/variant
            manifest=export_task(location,artifact,group,split='train')
            verify_export(location)
            learner_graph(artifact['learner'])
            variants.append({'variant':variant,'manifest':str((location/'manifest.json').relative_to(ROOT)),
                             'manifest_hash':sha256(location/'manifest.json')})
            counts['repair_variants']+=1
            if variant=='mixed':
                counts['missing_links']+=len(artifact['reference']['missing'])
                counts['wrong_links']+=len(artifact['reference']['wrong'])
                counts['fragmented_entities']+=len(artifact['reference']['fragmentations'])
        # Fixed recipe-derived cutoff; no target event is consulted to choose it.
        cutoff=args.jobs*recipe['arrival_interval']*.5
        prediction=modules.build_task('future_participation',execution,{'cutoff':cutoff,'horizon':20*recipe['service_scale']})
        location=output/episode_id/'prediction'
        export_task(location,prediction,group,split='train');verify_export(location)
        counts['world_events']+=len(execution['events'])
        counts['world_entities']+=len(execution['entities'])
        counts['genuine_exception_events']+=len(execution['reference']['exception_events'])
        counts['prediction_targets']+=len(prediction['reference']['future_events'])
        episodes.append({'id':episode_id,'group':group,'recipe':execution['provenance']['recipe'],
                         'execution_hash':fingerprint(execution),'projection':'flat' if flat else 'multi_object','variants':variants})
        if i%8==0: print(json.dumps({'completed':i+1,'requested':args.count,'seconds':time.monotonic()-start}),flush=True)
    report={'status':'generated_development_training_pilot','config':vars(args),'counts':dict(counts),
            'source_snapshot':source_snapshot,
            'train_compositions':TRAIN_COMPOSITIONS,'episodes':episodes,'seconds':time.monotonic()-start,
            'code_hashes':{str(p.relative_to(ROOT)):sha256(p) for folder in ['tgrepair','scripts'] for p in sorted((ROOT/folder).glob('*.py'))},
            'limits':['Symbolic execution, not a calibrated organizational simulator.',
                      'Family names change the schema vocabulary; they are not independent simulator implementations.',
                      'All artifacts are training/development, including the prediction reuse demonstration.',
                      'These datasets establish generation behavior, not synthetic-to-real learning benefit.']}
    (output/'dataset_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports'/f'{args.output}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'seconds':report['seconds'],'output':str(output)},indent=2))


if __name__=='__main__':main()
