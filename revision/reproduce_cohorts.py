"""Rebuild revision cohorts from user-located public sources and verify bytes.

This portability entry point leaves all frozen protocols and scientific adapters
unchanged. Absolute source paths in the original execution record need not exist
on a reproducer's machine; the supplied source must match the pinned checksum.
No repair model or outcome is evaluated.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision import cohorts
from tgrepair.data import anonymize, save_graph, sha256


def rebuild(domain, source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new reproduction directory: ' + str(output))
    prior = json.loads((ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json').read_text())
    native = json.loads((ROOT / 'revision/reports' / f'native_cohort_{domain}_v1.json').read_text())
    if sha256(source) != prior['source_sha256']:
        raise ValueError('The public input does not match the pinned source')
    if sha256(cohorts.__file__) != prior['adapter_sha256']:
        raise ValueError('The cohort adapter differs from the prepared experiment')
    plan = copy.deepcopy(json.loads((ROOT / 'revision/docs/COHORT_PLAN_V1.json').read_text()))
    amendment = json.loads((ROOT / 'revision/docs/NATIVE_COHORT_AMENDMENT_V1.json').read_text())
    plan['partition_sizes']['native_comparison'] = amendment['change']['revised_roots']
    graphs, bundles, selection, _, _ = getattr(cohorts, domain)(source, plan)
    output.mkdir(parents=True)
    report = {'status': 'rebuilding_revision_cohorts', 'domain': domain,
              'source_sha256': sha256(source), 'adapter_sha256': sha256(cohorts.__file__),
              'reproduction_script_sha256': sha256(__file__), 'partitions': {},
              'scope': 'Byte verification of source adaptation, partitioning and anonymization; no model outcomes opened.'}
    for part, graph in graphs.items():
        graph, mapping, _ = anonymize(graph, plan['anonymization_seeds'][part], return_maps=True)
        directory = output / part
        directory.mkdir()
        save_graph(graph, directory / 'reference_graph.json.gz')
        mapped = {mapping[r]: sorted(mapping[o] for o in members) for r, members in bundles[part].items()}
        (directory / 'bundles.json').write_text(json.dumps(mapped, sort_keys=True) + '\n')
        expected = native if part == 'native_comparison' else prior['partitions'][part]
        matches = {'graph': sha256(directory / 'reference_graph.json.gz') == expected['graph_sha256'],
                   'bundles': sha256(directory / 'bundles.json') == expected['bundles_sha256'],
                   'root_selection': selection['partitions'][part]['root_ids_sha256'] ==
                        (native['partitions'][part] if part == 'native_comparison' else expected)['root_ids_sha256']}
        report['partitions'][part] = {'matches': matches, 'events': len(graph.events), 'root_bundles': len(mapped)}
        if not all(matches.values()):
            (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
            raise AssertionError('Rebuilt partition differs: ' + part)
    report['status'] = 'all_revision_cohorts_reproduced_byte_identically'
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['bpi2017', 'bpi2019'], required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    rebuild(args.domain, args.source, args.output_directory)


if __name__ == '__main__':
    main()
