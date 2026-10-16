"""Score external history queries using view-independent native event labels."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,sha256
from tgrepair.application_queries import audit_queries


def main():
    destination=ROOT/'reports/external_application_queries_v2.json'
    if destination.exists():raise FileExistsError(destination)
    report={'status':'development_application_query_audit','results':{},
            'correction':'Initial structural-only evaluation derived exception labels from hidden learner attributes, producing an empty positive set. History metrics and cancellation queries were unaffected. This audit obtains exception-event labels from the pinned native parent, outside decoding.',
            'scorer_hash':sha256(ROOT/'tgrepair/application_queries.py')}
    for domain,short in [('batteryvault','battery'),('claimstream','claim')]:
        source=ROOT/'data/external_development_v1'/domain/'mixed'
        original=load_graph(source/'observed.json.gz')
        reference=json.loads((source/'reference.json').read_text())
        types=set(json.loads((source/'task.json').read_text())['target_types'])
        exceptions={e.uid for e in original.events if e.attributes.get('is_exception') in {1,True,'1'}}
        result={'input_hash':sha256(source/'observed.json.gz'),'reference_hash':sha256(source/'reference.json'),
                'exception_events':len(exceptions),'damaged':audit_queries(original,reference,types,exception_event_ids=exceptions),
                'methods':{}}
        for observation in ['mixed','structural']:
            name=f'external_{short}_{observation}_v1';location=ROOT/'data/decoding'/name
            graph=load_graph(location/'repaired.json.gz');clusters=json.loads((location/'clusters.json').read_text())
            result['methods'][name]={'graph_hash':sha256(location/'repaired.json.gz'),
                                    'cluster_hash':sha256(location/'clusters.json'),
                                    'queries':audit_queries(graph,reference,types,clusters,exceptions)}
        report['results'][domain]=result
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({domain:{'before_exception_f1':row['damaged']['queries']['exception_evidence']['f1'],
                            'methods':{k:v['queries']['queries']['exception_evidence']['f1'] for k,v in row['methods'].items()}}
                      for domain,row in report['results'].items()}))


if __name__=='__main__':main()
