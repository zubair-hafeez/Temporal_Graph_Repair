"""Run a bounded extra-control suite when its CPU predecessor finishes."""
import json
from pathlib import Path
import time

from revision.extra_controls import evaluate

ROOT = Path(__file__).resolve().parents[1]


def run(domain):
    predecessor = ROOT / 'revision/reports' / f'development_reliability_{domain}_seed0_v1.json'
    start = time.monotonic()
    print(json.dumps({'queued_extra_control_suite': domain, 'waiting_for': str(predecessor)}), flush=True)
    while True:
        try:
            report = json.loads(predecessor.read_text())
            if report['status'] == 'completed_development_controls':
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.monotonic() - start > 6 * 3600:
            raise TimeoutError('Predecessor did not complete; no dependent work was started')
        time.sleep(20)
    evaluate(domain, 0, 'extra')
