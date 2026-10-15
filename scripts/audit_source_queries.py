"""Recompute operational query diagnostics from preserved development artifacts."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.application_queries import audit_queries
from tgrepair.data import load_graph,sha256


def main():
    destination=ROOT/'reports/source_application_queries_v2.json'
    if destination.exists():raise FileExistsError(destination)
    base=ROOT/'data/sequential_development_v1'
    reference=json.loads((base/'reference.json').read_text())
    paths={
        'damaged':(base/'observed.json.gz',None),
        'participant_repair':(base/'repaired.json.gz',None),
        'iterative_source':(ROOT/'data/iterative_source_v1/round2.json.gz',ROOT/'data/iterative_source_v1/round2_clusters.json'),
    }
    for name in ['source_decode_feature_v2','source_decode_source16_v3','source_decode_mixed16_v3']:
        location=ROOT/'data/decoding'/name
        paths[name]=(location/'repaired.json.gz',location/'clusters.json')
    report={'status':'development_operational_query_diagnostic','reference_hash':sha256(base/'reference.json'),
            'correction':'v1 called timestamp equality record correctness. v2 uses first/last timestamp correctness, the predicate actually measured. v1 is preserved.',
            'results':{},'inputs':{}}
    for name,(graph_path,clusters_path) in paths.items():
        graph=load_graph(graph_path)
        clusters=json.loads(clusters_path.read_text()) if clusters_path else None
        report['results'][name]=audit_queries(graph,reference,{'Application'},clusters)
        report['inputs'][name]={'graph':str(graph_path.relative_to(ROOT)),'graph_hash':sha256(graph_path),
                                'clusters_hash':sha256(clusters_path) if clusters_path else None}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['queries']['cancellation_evidence']['f1'] for k,v in report['results'].items()}))


if __name__=='__main__':main()
