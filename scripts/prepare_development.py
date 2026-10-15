"""Read-only public source conversion; final target datasets are not loaded."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import load_bpi2012, load_logistics, save_graph, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attributes', action='store_true')
    args = parser.parse_args()
    evidence = ROOT.parent / 'Temporal_Graph_Papers_2024-2026/analysis/new_idea_20261015/evidence'
    sources = {
        'logistics': evidence / 'container_logistics.json',
        'bpi2012': ROOT.parent / 'Operational_Temporal_Graphs/data/raw/revision_bpi2012/bpi2012.xes.gz',
    }
    graphs = {} if args.attributes else {'logistics': load_logistics(sources['logistics'])}
    graphs.update({'bpi2012_' + k + ('_attributes' if args.attributes else ''): v
                   for k, v in load_bpi2012(sources['bpi2012'], args.attributes).items()})
    report = {'status': 'development_only', 'schema': 'event_participation_v1',
              'source_hashes': {k: sha256(p) for k, p in sources.items()},
              'excluded_fields': ['source identifiers', 'trace container', 'trace attributes',
                                  'source row order', 'object attributes', 'object-object relations'],
              'limits': ['Logistics is simulated and exploratory.',
                         'BPI 2012 is a real-source development log, not a final target.',
                         'Registry and entity types are assumed available.',
                         'Object-object evidence will require a separate matched-input track.'],
              'graphs': {}}
    if args.attributes:
        report['included_attributes'] = ['RequestedAmount copied from the source trace attribute into each event record.']
        report['excluded_fields'].remove('trace attributes')
        report['limits'].append('Repeated nonidentifier amounts are legitimate evidence in this declared richer track; entity registry attributes are not supplied.')
    for key, graph in graphs.items():
        path = ROOT / 'data/development' / (key + '.json.gz')
        save_graph(graph, path)
        report['graphs'][key] = {**graph.summary(), 'file': str(path.relative_to(ROOT)),
                                 'sha256': sha256(path)}
    (ROOT / 'reports').mkdir(exist_ok=True)
    report_name = 'development_attributes.json' if args.attributes else 'development_data.json'
    (ROOT / 'reports' / report_name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {a: v[a] for a in ['events', 'objects', 'participation_links']}
                      for k, v in report['graphs'].items()}, indent=2))


if __name__ == '__main__':
    main()
