"""Continue a terminated confirmation at a completed graph boundary.

Operational continuation only: frozen inference code, candidates, model,
calibration, ordered inputs and metrics remain unchanged. Preserve the incoming
partial report before adding any results. No partially decoded graph is reused.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_edit_policy import compare, thresholds_from_validation
from tgrepair.data import sha256
from tgrepair.factory_plugin import learner_graph
from tgrepair.decoding import FeatureScorer, NeuralScorer
from tgrepair.metrics import traceability
from tgrepair.provenance import snapshot_sources
from tgrepair.view_decoding import select_fields, decode_with_view
from tgrepair.confirmation_contract import verify_model
from temporal_factory.core import verify_export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    destination = ROOT/'reports'/f'{args.report}.json'
    report = json.loads(destination.read_text())
    if report['status'] != 'running_generated_confirmation':
        raise ValueError('Only an unfinished generated confirmation can continue')
    config = report['config']
    protocol_path = ROOT/config['confirmation_protocol']
    protocol = json.loads(protocol_path.read_text())
    if report['protocol_hash'] != sha256(protocol_path):
        raise ValueError('Protocol changed')
    for path, digest in protocol['implementation_hashes'].items():
        if sha256(ROOT/path) != digest:
            raise ValueError('Frozen implementation changed: '+path)
    expected = protocol['generated_confirmation']
    if config['generated'] != expected['attribute_collection'] or config['episodes'] != expected['count']:
        raise ValueError('Generated test collection differs from protocol')
    if [config['rounds'], 16, 32, config['precision']] != expected['decoder_contract']:
        raise ValueError('Decoder options differ from protocol')
    if config['fault_family']!='original' or config['mode']!='joint' or config['attribute_view']!='all':
        raise ValueError('Continuation is restricted to the declared primary generated comparison')
    model_path = ROOT/'checkpoints'/(config['model']+('.pt' if config['kind']=='neural' else '.joblib'))
    calibration_name = config['calibration'] or config['model']
    verify_model(ROOT, protocol, {}, config['model'], model_path, calibration_name)
    if sha256(model_path) != report['model_hash']:
        raise ValueError('Model changed')
    calibration_path = ROOT/'data/edit_learning'/f'{calibration_name}_validation_scores.npz'
    if sha256(calibration_path) != report['calibration_cache_hash']:
        raise ValueError('Calibration changed')
    thresholds, calibration = thresholds_from_validation(calibration_name, config['precision'])
    if thresholds != report['thresholds'] or calibration != report['action_validation_calibration']:
        raise ValueError('Thresholds changed')
    folder = ROOT/'data/generated'/config['generated']
    manifest = json.loads((folder/'dataset_manifest.json').read_text())
    episodes = random.Random(919).sample(manifest['episodes'], config['episodes'])
    ordered = [(episode, variant) for episode in episodes for variant in ['mixed', 'clean']]
    original_results = deepcopy(report['results'])
    if len(original_results)>len(ordered): raise ValueError('Too many saved results')
    audit_folder = ROOT/'data/decoding'/config['output']
    for i, (episode, variant) in enumerate(ordered):
        verify_export(folder/episode['id']/variant)
        if i<len(original_results):
            row = original_results[i]
            if (row['episode'], row['group'], row['variant']) != (episode['id'], episode['group'], variant):
                raise ValueError('Completed prefix ordering changed')
            audit = json.loads((audit_folder/f"{episode['id']}_{variant}.json").read_text())
            if row['edit_count'] != audit['edit_count']:
                raise ValueError('Completed graph audit mismatch')
    previous_hash = sha256(destination)
    backup = ROOT/'reports/interrupted'/f'{args.report}_{previous_hash[:16]}.json'
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists(): backup.write_bytes(destination.read_bytes())
    if sha256(backup) != previous_hash: raise ValueError('Partial-report backup mismatch')
    continuation = {'script_hash': sha256(__file__), 'input_report_hash': previous_hash,
                    'preserved_report': str(backup.relative_to(ROOT)), 'completed_graphs_at_resume': len(original_results),
                    'manifest_hash': sha256(folder/'dataset_manifest.json'),
                    'source_snapshot': snapshot_sources(ROOT),
                    'reason': 'Prior process returned termination signal 143; continue only unfinished graph evaluations.'}
    report.setdefault('operational_continuations', []).append(continuation)
    torch.set_num_threads(4)
    scorer = NeuralScorer(model_path, config['device']) if config['kind']=='neural' else FeatureScorer(model_path)
    start = time.monotonic(); previous_seconds = report['seconds']
    with threadpool_limits(limits=4):
        for episode, variant in ordered[len(original_results):]:
            location = folder/episode['id']/variant
            learner = json.loads((location/'learner.json').read_text())
            reference = json.loads((location/'reference.json').read_text())
            graph = learner_graph(learner); types = set(learner['schema']['target_types'])
            baseline = traceability(graph, reference, types)
            fields, view_profile = select_fields(graph, types, config['attribute_view'], config['stability_threshold'])
            state, audit = decode_with_view(graph, types, scorer, thresholds, rounds=config['rounds'],
                                            mode=config['mode'], attribute_fields=fields)
            row = {'episode': episode['id'], 'group': episode['group'], 'variant': variant, 'projection': episode['projection'],
                   'before': baseline, 'after': compare(graph, state, reference, types), 'edit_count': audit['edit_count'],
                   'seconds': audit['seconds'], 'stopped': audit['stopped'], 'attribute_view': view_profile}
            (audit_folder/f"{episode['id']}_{variant}.json").write_text(json.dumps(audit)+'\n')
            report['results'].append(row); report['seconds'] = previous_seconds+time.monotonic()-start
            destination.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'completed_graphs': len(report['results']), 'required_graphs': len(ordered)}), flush=True)
    if report['results'][:len(original_results)] != original_results:
        raise ValueError('An existing result was changed')
    summary = {}
    for variant in ['mixed', 'clean']:
        rows = [r for r in report['results'] if r['variant']==variant]
        summary[variant] = {'mean_before_f1': float(np.mean([r['before']['f1'] for r in rows])),
                            'mean_after_f1': float(np.mean([r['after']['history']['f1'] for r in rows])),
                            'mean_seconds': float(np.mean([r['seconds'] for r in rows])),
                            'edits': sum(r['edit_count'] for r in rows)}
    if sha256(model_path) != report['model_hash']: raise ValueError('Model changed during evaluation')
    report['summary'] = summary; report['status'] = 'completed_generated_confirmation'
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
