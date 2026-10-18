"""Assemble a local, complete revision review copy after final evidence review.

The preserved original review copy supplies the bounded generator and earlier
study. This overlay adds the revised manuscript and its reproducibility inputs.
It never uploads, submits, or changes a source artifact.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def check_inventory(base, inventory):
    for name, entry in inventory.items():
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError('Unsafe inventory member')
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('Review member changed: ' + name)


def build(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    final_path = ROOT / 'revision/reports/final_revision_review_v1.json'
    final = json.loads(final_path.read_text())
    if final['status'] != 'ready_for_author_review':
        raise RuntimeError('Complete the evidence and manuscript review first')
    for name, digest in final['checked_files'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Reviewed artifact changed: ' + name)
    original = json.loads((base / 'REVIEW_MANIFEST.json').read_text())
    check_inventory(base, original['files'])
    files = {name: base / name for name in original['files']}
    files['ORIGINAL_REVIEW_MANIFEST.json'] = base / 'REVIEW_MANIFEST.json'
    excluded = {'.git', '__pycache__', '.ipynb_checkpoints'}

    def add_tree(path):
        for item in sorted(path.rglob('*')):
            if item.is_file() and not excluded.intersection(item.parts) and item.suffix not in {'.pyc', '.pending'}:
                if item.relative_to(ROOT).as_posix() == 'revision/docs/STATE.json':
                    continue  # Historical copied state; authoritative state is revision/STATE.json.
                files['Temporal_Graph_Repair/' + item.relative_to(ROOT).as_posix()] = item

    add_tree(ROOT / 'revision')
    add_tree(ROOT / 'tgrepair')
    for collection in ['edit_train_v2', 'edit_train_attributes_v3']:
        add_tree(ROOT / 'data/edit_learning' / collection)
    for path in (ROOT / 'data/edit_learning').glob('*validation_scores.npz'):
        files['Temporal_Graph_Repair/' + path.relative_to(ROOT).as_posix()] = path
    for name in ['RESUME.md', 'STATE.json', 'README.md', 'requirements.txt']:
        files['Temporal_Graph_Repair/' + name] = ROOT / name

    output.mkdir(parents=True)
    inventory = {}
    for name, source in sorted(files.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        shutil.copyfile(source, target)
        if before != sha256(source) or before != sha256(target):
            raise ValueError('Source changed during copying: ' + name)
        inventory[name] = {'bytes': target.stat().st_size, 'sha256': before}
    guide = output / 'START_HERE.txt'
    guide.write_text(
        'LOCAL AUTHOR-REVIEW PACKAGE\n\n'
        'Current paper: Temporal_Graph_Repair/revision/paper/main.pdf\n'
        'Reproduction guide: Temporal_Graph_Repair/revision/REPRODUCE.txt\n'
        'Final evidence review: Temporal_Graph_Repair/revision/reports/final_revision_review_v1.json\n'
        'The earlier paper/main.pdf is retained as the original study archive.\n\n'
        'Verify all included bytes from the package root:\n'
        '  python Temporal_Graph_Repair/scripts/verify_review_bundle.py --bundle .\n\n'
        'This local copy includes derived public operational graphs for evidence replay.\n'
        'It is not an externally released dataset or a new license grant.\n'
        'Raw source files are obtained separately under their source terms.\n'
        'The original bounded temporal generator dependency retains its historical\n'
        'sibling directory name; no RF data/model work is included.\n')
    inventory[guide.name] = {'bytes': guide.stat().st_size, 'sha256': sha256(guide)}
    manifest = {'status': 'local_revision_author_review_artifact',
                'builder_sha256': sha256(__file__), 'final_review_sha256': sha256(final_path),
                'original_review_manifest_sha256': sha256(base / 'REVIEW_MANIFEST.json'),
                'files': inventory, 'file_count': len(inventory),
                'total_bytes': sum(v['bytes'] for v in inventory.values()),
                'scope': 'Private local review copy including derived public records and saved training features. No external upload, redistribution authorization, submission or acceptance claim.'}
    (output / 'REVIEW_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    check_inventory(output, inventory)
    print(json.dumps({'output': str(output), 'files': len(inventory), 'bytes': manifest['total_bytes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-review-directory', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    build(args.original_review_directory, args.output_directory)
