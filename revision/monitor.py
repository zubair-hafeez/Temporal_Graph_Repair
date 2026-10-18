"""Read-only resource and progress report for this paper's active workers."""
import json
import os
from pathlib import Path


def status():
    root = Path(__file__).resolve().parents[1]
    processes = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            parts = (directory / 'cmdline').read_bytes().split(b'\0')
            args = [p.decode(errors='replace') for p in parts if p]
            if not args or 'python' not in Path(args[0]).name:
                continue
            if not any(x in args for x in ['coupled-confirmation', 'native-confirmation']):
                continue
            fields = {}
            for line in (directory / 'status').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ['State', 'VmRSS', 'VmSize', 'Threads']:
                    fields[key] = value.strip()
            stat = (directory / 'stat').read_text().split()
            fields.update(pid=int(directory.name), task=args[-5:], cpu_seconds=(int(stat[13]) + int(stat[14])) / os.sysconf('SC_CLK_TCK'))
            processes.append(fields)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    results = {'load_average': os.getloadavg(), 'workers': processes, 'completed': [], 'running': []}
    for path in sorted((root / 'revision/reports/coupled_confirmation_v1').glob('*.json')):
        report = json.loads(path.read_text())
        category = 'completed' if report['status'] == 'completed_coupled_confirmation' else 'running'
        results[category].append(path.stem)
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapFree:')):
            key, value = line.split(':', 1)
            results[key] = value.strip()
    print(json.dumps(results, indent=2))
