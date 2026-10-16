"""Rebuild the pinned check split from its public XES source alone."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import load_bpi2012, save_graph, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Exact public BPI 2012 XES gzip')
    parser.add_argument('--output', required=True, help='New path for the derived check graph')
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    provenance = Path(str(output)+'.provenance.json')
    if output.exists() or provenance.exists():
        raise FileExistsError('Use a new graph and provenance destination')
    protocol_path = ROOT/'docs/CONFIRMATION_PROTOCOL_V1.json'
    protocol = json.loads(protocol_path.read_text())
    development_path = ROOT/'reports/development_attributes.json'
    development = json.loads(development_path.read_text())
    if sha256(source) != development['source_hashes']['bpi2012']:
        raise ValueError('BPI 2012 input differs from the declared public source')
    if sha256(ROOT/'tgrepair/data.py') != protocol['adapter_hashes']['tgrepair/data.py']:
        raise ValueError('Frozen source adapter changed')
    graph = load_bpi2012(source, include_attributes=True)['check']
    save_graph(graph, output)
    actual = sha256(output)
    expected = protocol['sources']['bpi2012_check']['sha256']
    if actual != expected:
        raise ValueError('Rebuilt check bytes differ; preserve output for investigation')
    record = {'status': 'exact_public_source_split_reproduction',
              'source': str(source), 'source_sha256': sha256(source),
              'output': str(output), 'output_sha256': actual,
              'protocol_sha256': sha256(protocol_path),
              'development_report_sha256': sha256(development_path),
              'script_sha256': sha256(__file__), 'summary': graph.summary(),
              'scope': 'Exact existing source conversion; no fitting, new split selection or confirmation retuning.'}
    provenance.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
