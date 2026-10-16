"""Integrity boundary for the separately frozen native-task comparison."""
import json
from pathlib import Path

from tgrepair.data import sha256

ROOT = Path(__file__).resolve().parents[1]


def verify_protocol(domain, seed, missing_rate, epochs, max_roots):
    path = ROOT / 'revision/docs/NATIVE_CONFIRMATION_PROTOCOL_V1.json'
    protocol = json.loads(path.read_text())
    if protocol['status'] != 'frozen_before_native_outcomes':
        raise ValueError('Native protocol is not frozen')
    key = [domain, seed, missing_rate, epochs, max_roots]
    if key not in protocol['jobs']:
        raise ValueError('Undeclared native evaluation')
    for group in ['implementation', 'inputs', 'models']:
        for name, expected in protocol[group].items():
            if sha256(ROOT / name) != expected:
                raise ValueError('Frozen native dependency changed: ' + name)
    return protocol


def run_suite(domain, device):
    from revision.native_here import train_native
    from revision.native_matched_controls import evaluate_native
    for seed in [0, 1, 2]:
        verify_protocol(domain, seed, .3, 30, 2048)
        for kind, callback, status in [
            ('here_native', lambda: train_native(domain, seed, .3, 30, 2048, device),
             'completed_here_native_confirmation'),
            ('native_controls_confirmation', lambda: evaluate_native(domain, seed, .3, 2048),
             'completed_native_controls_confirmation')]:
            suffix = '_epochs30' if kind == 'here_native' else ''
            report = ROOT / 'revision/reports' / f'{kind}_{domain}_seed{seed}_missing30_roots2048{suffix}_v1.json'
            if report.exists():
                saved = json.loads(report.read_text())
                if saved['status'] != status or saved['protocol_sha256'] != sha256(ROOT / 'revision/docs/NATIVE_CONFIRMATION_PROTOCOL_V1.json'):
                    raise RuntimeError('Incomplete or changed native job requires explicit recovery: ' + str(report))
                print(json.dumps({'verified_completed_native_job': str(report)}), flush=True)
                continue
            callback()
    print(json.dumps({'native_suite_completed': domain}), flush=True)
