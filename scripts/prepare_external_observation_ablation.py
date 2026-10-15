"""Hide attributes while retaining the exact external records and injected faults."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,save_graph,Graph,Event,sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='external_development_v1')
    parser.add_argument('--output',default='external_structural_v1')
    args=parser.parse_args();out=ROOT/'data'/args.output
    if out.exists():raise FileExistsError(out)
    for domain in ['batteryvault','claimstream']:
        for variant in ['clean','mixed','pipeline']:
            source=ROOT/'data'/args.source/domain/variant
            graph=load_graph(source/'observed.json.gz')
            structural=Graph(graph.name+'_structural',graph.objects,[Event(e.uid,e.time,e.kind,e.participants,{}) for e in graph.events])
            location=out/domain/variant;location.mkdir(parents=True)
            save_graph(structural,location/'observed.json.gz')
            (location/'reference.json').write_bytes((source/'reference.json').read_bytes())
            task=json.loads((source/'task.json').read_text())
            task['observation_sha256']=sha256(location/'observed.json.gz')
            task['observation_track']='structural_only; all event attributes hidden; native actors remain context entities'
            task['parent_task_hash']=sha256(source/'task.json')
            task['reason']='Development ablation after native v1 transfer degradation; same exact records and faults, no outcome-based row selection'
            (location/'task.json').write_text(json.dumps(task,indent=2)+'\n')
    print('Prepared six structural-only diagnostic tasks with unchanged records, handles and reference faults.')


if __name__=='__main__':main()
