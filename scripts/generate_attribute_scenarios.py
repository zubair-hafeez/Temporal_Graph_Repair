"""Paired attribute scenarios over exactly the existing executions and faults."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.attribute_scenarios import PROFILES,generate_attribute_view
from tgrepair.factory_plugin import learner_graph
from tgrepair.data import sha256
from tgrepair.provenance import snapshot_sources
from temporal_factory.core import export_task,verify_export


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();source=ROOT/'data/generated'/args.source;out=ROOT/'data/generated'/args.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.monotonic();snapshot=snapshot_sources(ROOT)
    manifest=json.loads((source/'dataset_manifest.json').read_text());episodes=[];counts=Counter()
    for i,episode in enumerate(manifest['episodes']):
        clean_path=source/episode['id']/'clean'
        clean=learner_graph(json.loads((clean_path/'learner.json').read_text()))
        clean_reference=json.loads((clean_path/'reference.json').read_text())
        profile=random.Random(args.seed+i).choice(PROFILES)
        attributes,recipe=generate_attribute_view(clean,args.seed+i,profile)
        # Ancestry maps are builder-only. They align paired descriptions across
        # different anonymized clean/damaged exports, never enter the learner.
        by_origin={clean_reference['event_origin'][uid]:values for uid,values in attributes.items()}
        specs={spec['name']:('numeric' if spec['family']=='measurement' else 'categorical') for spec in recipe['additional_fields']}
        variants=[]
        for variant in ['mixed','clean']:
            original=source/episode['id']/variant
            learner=json.loads((original/'learner.json').read_text())
            reference=json.loads((original/'reference.json').read_text())
            metadata=json.loads((original/'metadata.json').read_text())
            for event in learner['events']:event['attributes']=dict(by_origin[reference['event_origin'][event['uid']]])
            old_schema=learner['schema']['attributes']
            learner['schema']['attributes']={**{k:v for k,v in old_schema.items() if k in recipe['retained_original_fields']},**specs}
            metadata['attribute_scenario']=recipe
            metadata['parent_manifest_hash']=sha256(original/'manifest.json')
            metadata['attribute_scenario_source_snapshot']=snapshot
            artifact={'task':'temporal_graph_repair','learner':learner,'reference':reference,'metadata':metadata}
            location=out/episode['id']/variant
            split=json.loads((original/'manifest.json').read_text())['split']
            export_task(location,artifact,episode['group'],split=split);verify_export(location)
            learner_graph(learner)
            variants.append({'variant':variant,'manifest':str((location/'manifest.json').relative_to(ROOT)),
                             'manifest_hash':sha256(location/'manifest.json')})
            counts['repair_variants']+=1
        row=deepcopy(episode);row['variants']=variants;row['attribute_scenario']=recipe
        episodes.append(row);counts[profile]+=1;counts['world_events']+=len(clean.events)
        if i%32==0:print(json.dumps({'completed':i+1,'seconds':time.monotonic()-start}),flush=True)
    report={'status':'paired_generated_attribute_scenarios','config':vars(args),'source_snapshot':snapshot,
            'parent_manifest_hash':sha256(source/'dataset_manifest.json'),'counts':dict(counts),
            'compositions':manifest['compositions'],'episodes':episodes,'seconds':time.monotonic()-start,
            'limits':['Same exact worlds, visible types, records and fault labels as parent collection.',
                      'Only simulated descriptive/measurement fields vary; no new target identity copies.',
                      'Paired derivatives retain parent execution groups and cannot be split independently.',
                      'Future-participation artifacts are not re-exported in this repair-only ablation.']}
    (out/'dataset_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports'/f'{args.output}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
