"""Build reserved tasks only after the concrete protocol has been frozen."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.confirmation_sources import load_bpi2017_confirmation,load_bpi2019_confirmation,structural_view
from tgrepair.corruption import mixed_faults
from tgrepair.data import load_graph,save_graph,sha256
from tgrepair.external_ocel import load_external_ocel
from tgrepair.pipeline_faults import pipeline_faults
from tgrepair.provenance import snapshot_sources


def clean_reference(graph):
    return {'event_participation':{e.uid:list(e.participants) for e in graph.events},
            'identity':{o:o for o in graph.objects},'event_origin':{e.uid:e.uid for e in graph.events},
            'missing':[],'wrong':[],'fragmentations':[],'config':{'faults':'none'}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--domain',required=True)
    args=parser.parse_args()
    protocol_path=ROOT/args.protocol;protocol=json.loads(protocol_path.read_text())
    if protocol['status']!='frozen_before_reserved_evaluation':raise ValueError('Final protocol is not frozen')
    specification=protocol['sources'][args.domain]
    for name,digest in protocol['adapter_hashes'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Adapter differs from frozen protocol: '+name)
    source=Path(specification['path'])
    if not source.is_absolute():source=ROOT/source
    if sha256(source)!=specification['sha256']:raise ValueError('Reserved source hash mismatch')
    output=ROOT/protocol['task_root']/args.domain
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    start=time.monotonic();snapshot=snapshot_sources(ROOT)
    if args.domain=='bpi2017':
        graph,audit=load_bpi2017_confirmation(source,**specification['adapter_options'])
    elif args.domain=='bpi2019':
        graph,audit=load_bpi2019_confirmation(source,**specification['adapter_options'])
    elif args.domain=='bpi2012_check':
        graph=load_graph(source);audit={'summary':graph.summary(),'condition':'denormalized requested amount; same-process check split'}
    else:
        graph,audit=load_external_ocel(source,args.domain+'_confirmation_native',**specification['adapter_options'])
    save_graph(graph,output/'available_window.json.gz')
    (output/'adapter_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    report={'status':'prepared_reserved_confirmation_tasks','protocol_hash':sha256(protocol_path),
            'domain':args.domain,'source_sha256':sha256(source),'source_snapshot':snapshot,
            'adapter_audit':audit,'tasks':[]}
    types=set(specification['target_types'])
    if not types<=set(graph.objects.values()):raise ValueError('Target type absent in selected cohort')
    exception_origins={e.uid for e in graph.events if e.attributes.get('is_exception') in {True,1,'1'}}
    for variant in ['clean',*protocol['fault_families']]:
        for seed in ([0] if variant=='clean' else protocol['fault_seeds']):
            if variant=='clean':observation,reference=graph,clean_reference(graph)
            elif variant=='mixed':observation,reference=mixed_faults(graph,seed,types)
            elif variant=='pipeline':observation,reference=pipeline_faults(graph,seed,types)
            else:raise ValueError('Unknown predeclared fault family')
            reference['exception_events']=[uid for uid,origin in reference['event_origin'].items() if origin in exception_origins]
            for view in specification['views']:
                visible=structural_view(observation) if view=='structural' else observation
                folder=output/view/f'{variant}_{seed}'
                folder.mkdir(parents=True)
                save_graph(visible,folder/'observed.json.gz')
                (folder/'reference.json').write_text(json.dumps(reference,separators=(',',':'))+'\n')
                task={'domain':args.domain,'variant':variant,'fault_seed':seed,'observation_view':view,
                      'target_types':sorted(types),'source_sha256':sha256(source),
                      'protocol_hash':sha256(protocol_path),'adapter_snapshot':snapshot,
                      'observation_sha256':sha256(folder/'observed.json.gz'),
                      'reference_sha256':sha256(folder/'reference.json')}
                (folder/'task.json').write_text(json.dumps(task,indent=2)+'\n')
                report['tasks'].append({'directory':str(folder.relative_to(ROOT)),**task})
    report['seconds']=time.monotonic()-start
    (ROOT/'reports'/f'confirmation_prepare_{args.domain}_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'domain':args.domain,'tasks':len(report['tasks']),'summary':graph.summary(),'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
