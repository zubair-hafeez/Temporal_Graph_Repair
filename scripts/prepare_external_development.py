"""Prepare only the predeclared independent-simulator development domains."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import save_graph, sha256
from tgrepair.external_ocel import load_external_ocel
from tgrepair.corruption import mixed_faults
from tgrepair.pipeline_faults import pipeline_faults
from tgrepair.provenance import snapshot_sources
from tgrepair.metrics import traceability

DEVELOPMENT_TYPES = {'batteryvault': ['BatteryPack', 'ServiceOrder'],
                     'claimstream': ['Claim', 'Exposure']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='external_development_v1')
    args = parser.parse_args()
    directory = ROOT/'data/external/relational_ocel'
    output = ROOT/'data'/args.output
    if output.exists():
        raise FileExistsError(output)
    start = time.monotonic()
    snapshot = snapshot_sources(ROOT)
    metadata = json.loads((directory/'record_metadata.json').read_text())
    if metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('Unexpected source license')
    checksums = {name: digest for digest, name in
                 (line.split() for line in (directory/'SHA256SUMS').read_text().splitlines())}
    md5s = {entry['key']: entry['checksum'] for entry in metadata['files']}
    report = {'status': 'development_sources_only', 'source_snapshot': snapshot,
              'doi': metadata['doi'], 'license': 'CC BY 4.0',
              'protocol_hash': sha256(ROOT/'docs/EXTERNAL_WORKFLOW_PROTOCOL.md'),
              'limits': ['Independent simulator, not real operational deployments.',
                         'Window-censored histories; native event fields only.',
                         'Final four external scenarios remain unopened.'], 'datasets': {}}
    for name, types in DEVELOPMENT_TYPES.items():
        source = directory/(name+'.jsonocel.gz')
        digest = sha256(source)
        if digest != checksums[source.name]:
            raise ValueError(f'SHA256 mismatch: {source.name}')
        if 'md5:'+hashlib.md5(source.read_bytes()).hexdigest() != md5s[source.name]:
            raise ValueError(f'MD5 mismatch: {source.name}')
        clean, audit = load_external_ocel(source, name+'_native_window', seed=61001)
        if not set(types) <= set(clean.objects.values()):
            raise ValueError('Declared target type absent from window')
        row = {'source_sha256': digest, 'audit': audit, 'target_types': types, 'tasks': {}}
        for variant in ['clean', 'mixed', 'pipeline']:
            if variant == 'clean':
                graph = clean
                reference = {'event_participation': {e.uid:list(e.participants) for e in graph.events},
                             'identity': {o:o for o in graph.objects},
                             'event_origin': {e.uid:e.uid for e in graph.events},
                             'missing': [], 'wrong': [], 'fragmentations': [], 'config': {'faults':'none'}}
            elif variant == 'mixed':
                graph, reference = mixed_faults(clean, 61001, set(types))
            else:
                graph, reference = pipeline_faults(clean, 61001, set(types))
            location = output/name/variant
            location.mkdir(parents=True)
            save_graph(graph, location/'observed.json.gz')
            (location/'reference.json').write_text(json.dumps(reference, separators=(',',':'))+'\n')
            task = {'domain':name,'variant':variant,'target_types':types,
                    'observation_sha256':sha256(location/'observed.json.gz'),
                    'reference_sha256':sha256(location/'reference.json'),
                    'source_sha256':digest,'adapter_snapshot':snapshot}
            (location/'task.json').write_text(json.dumps(task,indent=2)+'\n')
            row['tasks'][variant] = {'location':str(location.relative_to(ROOT)),
                                    'before':traceability(graph,reference,set(types)),
                                    'fault_counts':{key:len(reference[key]) for key in ['missing','wrong','fragmentations']},
                                    **task}
        report['datasets'][name] = row
        print(json.dumps({'domain':name,'events':len(clean.events),'objects':len(clean.objects),
                          'faults':{k:v['fault_counts'] for k,v in row['tasks'].items()}}),flush=True)
    report['seconds'] = time.monotonic()-start
    (ROOT/'reports'/f'{args.output}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
