"""Create an explicit path-only copy of the original confirmation protocol."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-map', required=True, help='JSON object: source domain to existing local file')
    parser.add_argument('--output', required=True)
    parser.add_argument('--task-root', default='data/reproduction_confirmation_v1')
    args = parser.parse_args()
    original_path = ROOT/'docs/CONFIRMATION_PROTOCOL_V1.json'
    original = json.loads(original_path.read_text())
    output = ROOT/args.output
    if output.exists():
        raise FileExistsError(output)
    mapping_path = Path(args.source_map)
    mapping = json.loads(mapping_path.read_text())
    if not mapping or not set(mapping) <= set(original['sources']):
        raise ValueError('Map must contain known predeclared source domains')
    modified = copy.deepcopy(original)
    changes = []
    for domain, value in mapping.items():
        source = Path(value).expanduser().resolve()
        if sha256(source) != original['sources'][domain]['sha256']:
            raise ValueError('Different input bytes for '+domain)
        modified['sources'][domain]['path'] = str(source)
        changes.append({'field': 'sources.'+domain+'.path', 'before': original['sources'][domain]['path'], 'after': str(source)})
    task_root = Path(args.task_root)
    if task_root.is_absolute() or '..' in task_root.parts or (ROOT/task_root).exists():
        raise ValueError('Use a new relative task directory')
    modified['task_root'] = task_root.as_posix()
    changes.append({'field': 'task_root', 'before': original['task_root'], 'after': task_root.as_posix()})
    # Assert that no model, source checksum, cohort, fault, calibration or decoder
    # setting changed while relocating the artifacts.
    check = copy.deepcopy(modified)
    check['task_root'] = original['task_root']
    for domain in mapping:
        check['sources'][domain]['path'] = original['sources'][domain]['path']
    assert check == original
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(modified, indent=2)+'\n')
    record = {'status': 'path_only_reproduction_protocol', 'original_protocol_sha256': sha256(original_path),
              'reproduction_protocol_sha256': sha256(output), 'source_map_sha256': sha256(mapping_path),
              'script_sha256': sha256(__file__), 'changes': changes,
              'limit': 'Historical status/date are retained verbatim. This copy is a reproduction, not another prospective freeze.'}
    output.with_suffix('.relocation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
