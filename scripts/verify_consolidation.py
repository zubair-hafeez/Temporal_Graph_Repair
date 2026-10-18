"""Check that every file from the retired review copies remains available."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    receipt = json.loads((ROOT / 'archives/CONSOLIDATION_MANIFEST.json').read_text())
    checked = {}
    for entry in receipt['entries']:
        if entry.get('removed_by_user_request'):
            continue
        path = (ROOT / entry['destination']).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Destination outside consolidated project')
        if path not in checked:
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            checked[path] = (path.stat().st_size, h.hexdigest())
        if checked[path] != (entry['bytes'], entry['sha256']):
            raise ValueError('Retained file changed: ' + entry['destination'])
    print(json.dumps({'status': 'all_nonretired_review_files_preserved',
                      'original_file_entries': len(receipt['entries']),
                      'earlier_manuscript_entries_removed_by_user': sum(bool(e.get('removed_by_user_request')) for e in receipt['entries']),
                      'distinct_retained_files': len(checked)}))


if __name__ == '__main__':
    main()
