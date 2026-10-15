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
VALIDATION_COMPOSITIONS = [
    ['queues','branch_join','handoff','retry'],
    ['queues','batch_split','handoff','cancel_return'],
    ['queues','branch_join','batch_split','cancel_return'],
]
TEST_COMPOSITIONS = [
    ['queues','branch_join','batch_split','handoff','retry','cancel_return'],
    ['queues','branch_join','batch_split','handoff','retry'],
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--jobs',type=int,default=12)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--output',default='workflow_pilot_v1')
    p.add_argument('--split',choices=['train','validation','test'],default='train')
    p.add_argument('--diversity',choices=['original','expanded'],default='original')
    args=p.parse_args()
    if not 1<=args.count<=2000: raise ValueError('Pilot generation count must be in [1,2000]')
    output=ROOT/'data/generated'/args.output
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    source_snapshot=snapshot_sources(ROOT)
    modules=registry();rng=random.Random(args.seed)
    compositions={'train':TRAIN_COMPOSITIONS,'validation':VALIDATION_COMPOSITIONS,'test':TEST_COMPOSITIONS}[args.split]
    counts=Counter();episodes=[];start=time.monotonic()
    for i in range(args.count):
        mechanisms=compositions[i%len(compositions)]
        recipe={'seed':args.seed+i,'family':['service','fulfillment','assembly'][(i//len(compositions))%3],
                'jobs':args.jobs,'max_children':rng.choice([1,2,3,4]),'resources':rng.choice([2,4,8]),
                'arrival_interval':rng.choice([.5,2.,8.]),'service_scale':rng.choice([.5,1.,3.]),
                'attribute_palette':rng.choice([2,4,8,16]),'mechanisms':mechanisms}
        if args.diversity=='expanded':
            recipe.update({'work_stages':rng.choice([0,2,6,12,24]),'work_repetitions':rng.choice([1,2,3]),
                           'lifecycle_records':rng.choice([False,True]),'waiting_scale':rng.choice([1.,10.,100.]),
                           'resources':rng.choice([4,8,16]),'jobs':rng.choice([12,24,48]),
                           'attribute_palette':rng.choice([2,8,16,32,64])})
        execution=modules.create_world('workflow',recipe).run()
        roles=execution['schema']['entity_roles']
        flat = (i//(3*len(compositions)))%2 == 0
        visible_types = [roles['child'],roles['resource']] if flat else execution['schema']['entity_types']
        target_types = [roles['child']] if flat else [roles['parent'],roles['child']]
        visible_attributes = (rng.choice([[],['Amount'],['Amount'],['Amount','Priority','Category']])
                              if args.diversity=='expanded' else ['Amount','Priority','Category'])
        episode_id=f'episode-{i:06d}'
        group=fingerprint({'recipe':execution['provenance']['recipe_hash'],'source':execution['provenance']['source_hash']})
        variants=[]
        for variant in ['mixed','clean']:
            config={'target_types':target_types,'observation_seed':args.seed+100000+i,
                    'visible_types':visible_types,
                    'visible_attributes':visible_attributes,
                    'missing_rate':rng.choice([.05,.1,.2]) if variant=='mixed' else 0.,
                    'wrong_rate':rng.choice([.02,.1,.2]) if variant=='mixed' else 0.,
                    'fragment_rate':rng.choice([.1,.25,.5]) if variant=='mixed' else 0.,'gap_mode':'contiguous'}
            artifact=modules.build_task('temporal_graph_repair',execution,config)
            location=output/episode_id/variant
            manifest=export_task(location,artifact,group,split=args.split)
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
        cutoff=recipe['jobs']*recipe['arrival_interval']*.5
        prediction=modules.build_task('future_participation',execution,{'cutoff':cutoff,'horizon':20*recipe['service_scale']})
        location=output/episode_id/'prediction'
        export_task(location,prediction,group,split=args.split);verify_export(location)
        counts['world_events']+=len(execution['events'])
        counts['world_entities']+=len(execution['entities'])
        counts['genuine_exception_events']+=len(execution['reference']['exception_events'])
        counts['prediction_targets']+=len(prediction['reference']['future_events'])
        episodes.append({'id':episode_id,'group':group,'recipe':execution['provenance']['recipe'],
                         'execution_hash':fingerprint(execution),'projection':'flat' if flat else 'multi_object','variants':variants})
        if i%8==0: print(json.dumps({'completed':i+1,'requested':args.count,'seconds':time.monotonic()-start}),flush=True)
    report={'status':'generated_development_workflow_collection','config':vars(args),'counts':dict(counts),
            'source_snapshot':source_snapshot,
            'compositions':compositions,'episodes':episodes,'seconds':time.monotonic()-start,
            'code_hashes':{str(p.relative_to(ROOT)):sha256(p) for folder in ['tgrepair','scripts'] for p in sorted((ROOT/folder).glob('*.py'))},
            'limits':['Symbolic execution, not a calibrated organizational simulator.',
                      'Family names change the schema vocabulary; they are not independent simulator implementations.',
                      'The split is a generated learning/development partition, not a real-source final test.',
                      'These datasets establish generation behavior, not synthetic-to-real learning benefit.']}
    (output/'dataset_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports'/f'{args.output}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'seconds':report['seconds'],'output':str(output)},indent=2))


if __name__=='__main__':main()
