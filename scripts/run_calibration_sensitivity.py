"""Post-freeze source-model calibration sensitivity; not a primary comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1..4 workers')
    declaration_path = ROOT/'docs/CALIBRATION_SENSITIVITY_V1.json'
    declaration = json.loads(declaration_path.read_text())
    protocol_path = ROOT/'docs/CONFIRMATION_PROTOCOL_V1.json'
    if sha256(protocol_path) != declaration['original_protocol_hash']:
        raise ValueError('Original protocol changed')
    destination = ROOT/'reports/calibration_sensitivity_v1.json'
    lock_path = ROOT/'data/.calibration_sensitivity_v1.lock'
    def execute(job):
        output = ROOT/'reports'/f"{job['output']}.json"
        command = [sys.executable, str(ROOT/'scripts/evaluate_external_policy.py'),
                   '--task', job['task'], '--model', declaration['model'], '--kind', 'feature',
                   '--calibration', declaration['calibration'], '--protocol', str(protocol_path.relative_to(ROOT)),
                   '--output', job['output']]
        if output.exists():
            if not args.resume:
                raise FileExistsError(output)
            previous = json.loads(output.read_text())
            if previous['pins']['protocol_hash'] != declaration['original_protocol_hash']:
                raise ValueError('Different frozen implementation protocol')
            if previous['status'] == 'completed_confirmation_evaluation':
                return {'output': job['output'], 'status': 'already_completed', 'report_hash': sha256(output)}
            command.append('--resume')
        start = time.monotonic()
        with (ROOT/'reports'/f"{job['output']}.log").open('a' if args.resume else 'x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        if result['status'] != 'completed_confirmation_evaluation':
            raise RuntimeError('Incomplete sensitivity run')
        return {'output': job['output'], 'status': 'completed', 'report_hash': sha256(output),
                'command': command, 'seconds': time.monotonic()-start, 'f1': result['after']['history']['f1']}
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists() and not args.resume:
            raise FileExistsError(destination)
        report = {'status': 'running_post_freeze_calibration_sensitivity',
                  'declaration_hash': sha256(declaration_path), 'script_hash': sha256(__file__),
                  'jobs': declaration['jobs'], 'outcomes': [], 'scope': declaration['scope']}
        destination.write_text(json.dumps(report, indent=2)+'\n')
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(execute, job): job for job in declaration['jobs']}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as error:
                    row = {'output': futures[future]['output'], 'status': 'failed', 'error': repr(error)}
                report['outcomes'].append(row)
                destination.write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps(row), flush=True)
        report['status'] = ('completed_post_freeze_calibration_sensitivity'
                            if all(x['status'] != 'failed' for x in report['outcomes']) else 'sensitivity_has_operational_failures')
        destination.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
