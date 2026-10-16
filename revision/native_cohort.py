"""Extend the reserved native cohort without changing other partitions."""
import copy
import json
from pathlib import Path

from revision import cohorts
from tgrepair.data import anonymize, save_graph, sha256

ROOT = Path(__file__).resolve().parents[1]


def prepare(domain):
    amendment_path = ROOT / 'revision/docs/NATIVE_COHORT_AMENDMENT_V1.json'
    amendment = json.loads(amendment_path.read_text())
    base_path = ROOT / amendment['base_plan']
    plan = copy.deepcopy(json.loads(base_path.read_text()))
    plan['partition_sizes']['native_comparison'] = amendment['change']['revised_roots']
    destination = ROOT / 'revision/data/native_cohort_v1' / domain
    if destination.exists():
        raise FileExistsError(destination)
    prior_path = ROOT / 'revision/reports' / f'cohorts_{domain}_v1.json'
    prior = json.loads(prior_path.read_text())
    source = Path(prior['source_path'])
    if sha256(source) != prior['source_sha256']:
        raise ValueError('Source checksum changed')
    graphs, bundles, audit, types, root_type = getattr(cohorts, domain)(source, plan)
    for part in ['development', 'annotation', 'confirmation']:
        if audit['partitions'][part]['root_ids_sha256'] != prior['partitions'][part]['root_ids_sha256']:
            raise AssertionError('Amendment changed an earlier root partition')
    # Recompute the original final prefix independently: the extended native
    # graph may change anonymized handles, but its raw selection must be nested.
    # assign_partitions sorts the same eligible pool and extends only its last
    # block; this is additionally tested on a toy pool.
    graph = graphs['native_comparison']
    graph, mapping, _ = anonymize(graph, amendment['anonymization_seed'], return_maps=True)
    mapped = {mapping[r]: sorted(mapping[o] for o in members)
              for r, members in bundles['native_comparison'].items()}
    if len(mapped) != amendment['change']['revised_roots']:
        raise AssertionError('Native root has no retained history')
    destination.mkdir(parents=True)
    save_graph(graph, destination / 'reference_graph.json.gz')
    (destination / 'bundles.json').write_text(json.dumps(mapped, sort_keys=True) + '\n')
    audit.update({'status': 'expanded_native_cohort_prepared_outcomes_unopened',
                  'domain': domain, 'target_types': sorted(types), 'root_type': root_type,
                  'amendment_sha256': sha256(amendment_path), 'base_plan_sha256': sha256(base_path),
                  'previous_cohort_manifest_sha256': sha256(prior_path),
                  'source_sha256': prior['source_sha256'],
                  'implementation_sha256': sha256(__file__), 'adapter_sha256': sha256(cohorts.__file__),
                  'graph_sha256': sha256(destination / 'reference_graph.json.gz'),
                  'bundles_sha256': sha256(destination / 'bundles.json'),
                  'summary': graph.summary(), 'earlier_partitions_unchanged': True,
                  'outcomes_opened': False})
    report = ROOT / 'revision/reports' / f'native_cohort_{domain}_v1.json'
    report.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'domain': domain, 'roots': len(mapped), 'events': len(graph.events),
                      'report': str(report)}), flush=True)
