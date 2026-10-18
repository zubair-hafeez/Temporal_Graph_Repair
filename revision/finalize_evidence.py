"""Wait for the frozen suite, then audit artifacts and generate all tables.

This never selects models, edits thresholds or writes manuscript conclusions.
It leaves publication-readiness decisions to the final evidence review.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.confirmation import verify_protocol
from tgrepair.data import sha256


def main():
    protocol = verify_protocol()
    files = ['revision/audit_coupled_results.py', 'revision/build_coupled_results.py']
    pins = {name: sha256(ROOT / name) for name in files}
    status_path = ROOT / 'revision/reports/evidence_finalization_v1.json'
    start = time.monotonic()
    previous = None
    while time.monotonic() - start < 24 * 3600:
        completed = 0
        for job in protocol['jobs']:
            path = ROOT / 'revision/reports/coupled_confirmation_v1' / (job['id'] + '.json')
            if path.exists():
                report = json.loads(path.read_text())
                completed += int(report['status'] == 'completed_coupled_confirmation')
        if completed != previous:
            status = {'status': 'waiting_for_all_frozen_jobs', 'completed': completed,
                      'declared': len(protocol['jobs']), 'postprocessing_pins': pins}
            status_path.write_text(json.dumps(status, indent=2) + '\n')
            print(json.dumps(status), flush=True)
            previous = completed
        if completed == len(protocol['jobs']):
            break
        time.sleep(30)
    else:
        raise TimeoutError('Frozen suite did not finish within the 24-hour monitoring window')
    verify_protocol()
    for name, digest in pins.items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Postprocessing implementation changed while waiting')
    try:
        for name in files:
            subprocess.run([sys.executable, name], cwd=ROOT, check=True)
    except Exception as error:
        status['status'] = 'postprocessing_failed_requires_review'
        status['error'] = repr(error)
        status_path.write_text(json.dumps(status, indent=2) + '\n')
        raise
    status['status'] = 'complete_audited_evidence_and_tables_ready_for_manuscript_review'
    status['seconds'] = time.monotonic() - start
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
