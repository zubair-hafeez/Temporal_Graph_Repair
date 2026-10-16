"""Reproduce the paired training comparison with independent executed worlds.

This is development replication, not permission to inspect reserved targets.
Each finished subprocess is recorded with its command and artifact hash. A
resume validates those artifacts rather than silently skipping an existing file.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256
from tgrepair.provenance import snapshot_sources


def stages(replica):
    seed = 70000 + replica * 100000
    train = f'workflow_train_repeat{replica}'
    valid = f'workflow_validation_repeat{replica}'
    atrain, aval = train + '_attributes', valid + '_attributes'
    plain_cache = f'edit_train_plain_repeat{replica}'
    train_cache = f'edit_train_attributes_repeat{replica}'
    valid_cache = f'edit_validation_attributes_repeat{replica}'
    return [
        (train, ['generate_workflow_pilot.py', '--count', '512', '--seed', str(seed), '--output', train, '--split', 'train', '--diversity', 'expanded']),
        (valid, ['generate_workflow_pilot.py', '--count', '96', '--seed', str(seed + 20000), '--output', valid, '--split', 'validation', '--diversity', 'expanded']),
        (atrain, ['generate_attribute_scenarios.py', '--source', train, '--output', atrain, '--seed', str(321000 + replica * 200000)]),
        (aval, ['generate_attribute_scenarios.py', '--source', valid, '--output', aval, '--seed', str(421000 + replica * 200000)]),
        (plain_cache, ['prepare_edit_learning.py', '--generated', train, '--episodes', '512', '--output', plain_cache, '--partial']),
        (train_cache, ['prepare_edit_learning.py', '--generated', atrain, '--episodes', '512', '--output', train_cache, '--partial']),
        (valid_cache, ['prepare_edit_learning.py', '--generated', aval, '--episodes', '96', '--output', valid_cache]),
        (f'edit_feature_plain_repeat{replica}', ['train_edit_feature_policy.py', '--train', plain_cache, '--validation', valid_cache, '--output', f'edit_feature_plain_repeat{replica}']),
        (f'edit_feature_attributes_repeat{replica}', ['train_edit_feature_policy.py', '--train', train_cache, '--validation', valid_cache, '--output', f'edit_feature_attributes_repeat{replica}']),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicate', type=int, choices=[1, 2], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    destination = ROOT/'reports'/f'training_repeat{args.replicate}_pipeline.json'
    if destination.exists() != args.resume:
        raise ValueError('Existing pipeline requires --resume; new pipeline must omit it')
    commands = stages(args.replicate)
    script_pins = {name: sha256(ROOT/'scripts'/name)
                   for name in sorted({stage[1][0] for stage in commands})}
    if args.resume:
        report = json.loads(destination.read_text())
        if report['script_pins'] != script_pins:
            raise ValueError('Pipeline script changed during replication')
    else:
        report = {'status': 'running_development_training_replication',
                  'replicate': args.replicate, 'script_pins': script_pins,
                  'source_snapshot': snapshot_sources(ROOT), 'stages': []}
    for name, arguments in commands:
        artifact = ROOT/'reports'/f'{name}.json'
        completed = next((s for s in report['stages'] if s['name'] == name), None)
        if completed is not None:
            if sha256(artifact) != completed['artifact_hash']:
                raise ValueError('Completed stage artifact changed')
            continue
        command = [sys.executable, str(ROOT/'scripts'/arguments[0]), *arguments[1:]]
        report['active_stage'] = name
        destination.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'stage': name, 'status': 'started'}), flush=True)
        start = time.monotonic()
        with (ROOT/'reports'/f'{name}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        row = {'name': name, 'command': command, 'seconds': time.monotonic()-start,
               'artifact_hash': sha256(artifact)}
        report['stages'].append(row)
        destination.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'stage': name, 'status': 'completed', 'seconds': row['seconds']}), flush=True)
    report.pop('active_stage', None)
    report['status'] = 'completed_development_training_replication'
    destination.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
