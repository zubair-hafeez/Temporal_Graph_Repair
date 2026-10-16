"""Verify copied artifact bytes before running any bundled model or analysis."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    args = parser.parse_args()
    root = Path(args.bundle).resolve()
    manifest = json.loads((root/'REVIEW_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Unsafe manifest path')
        content = path.read_bytes()
        if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
            raise ValueError('Artifact changed or missing: '+name)
    print(json.dumps({'status': 'copied_artifact_checksums_passed', 'files': len(manifest['files']),
                      'artifact_status': manifest['status'],
                      'limit': 'Byte integrity does not itself establish scientific validity.'}))


if __name__ == '__main__':
    main()
