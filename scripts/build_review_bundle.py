"""Build a local review artifact; never upload, publish, or alter frozen evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--allow-incomplete', action='store_true', help='Internal portability check only')
    args = parser.parse_args()
    destination = Path(args.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    evidence = json.loads((ROOT/'reports/confirmation_real_evidence_v1.json').read_text())
    if not args.allow_incomplete and evidence['status'] != 'complete_selected_confirmation':
        raise ValueError('Final review artifact requires every selected confirmation comparison')
    if not args.allow_incomplete:
        generated = json.loads((ROOT/'reports/generated_confirmation_evidence_v1.json').read_text())
        if generated['status'] != 'complete_generated_confirmation_analysis':
            raise ValueError('Generated confirmation is incomplete')
        if (ROOT/'docs/CALIBRATION_SENSITIVITY_V1.json').exists():
            sensitivity = json.loads((ROOT/'reports/calibration_sensitivity_v1.json').read_text())
            if sensitivity['status'] != 'completed_post_freeze_calibration_sensitivity':
                raise ValueError('Declared follow-up comparison is incomplete')
    protocol_path = ROOT/'docs/CONFIRMATION_PROTOCOL_V1.json'
    protocol = json.loads(protocol_path.read_text())
    implementation = (protocol['implementation_hashes'] | protocol['adapter_hashes'] |
                      protocol['generated_confirmation']['script_hashes'] |
                      protocol['target_local_adaptation']['script_hashes'])
    for name, digest in implementation.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen inference implementation changed: '+name)
    for name, model in protocol['models'].items():
        suffix = '.pt' if model['kind'] == 'neural' else '.joblib'
        if sha256(ROOT/'checkpoints'/(name+suffix)) != model['sha256']:
            raise ValueError('Frozen model changed: '+name)
        for calibration, digest in model['calibrations'].items():
            if sha256(ROOT/'data/edit_learning'/(calibration+'_validation_scores.npz')) != digest:
                raise ValueError('Frozen calibration changed: '+calibration)
    files = {}
    def add(path, relative=None):
        path = Path(path)
        relative = relative or Path('Temporal_Graph_Repair')/path.relative_to(ROOT)
        if relative.as_posix() in files and files[relative.as_posix()] != path:
            raise ValueError('Conflicting artifact path')
        files[relative.as_posix()] = path
    for name in ['README.md', 'requirements.txt', 'STATE.json', 'RESUME.md', 'LEDGER.md']:
        add(ROOT/name)
    for folder in ['tgrepair', 'scripts', 'tests', 'docs', 'paper', 'reports', 'checkpoints']:
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in {'.log', '.aux', '.out', '.pyc'}:
                continue
            add(path)
    for path in sorted((ROOT/'data/edit_learning').glob('*validation_scores.npz')):
        add(path)
    for path in sorted((ROOT/'data/source_snapshots').glob('*.tar.gz')):
        add(path)
    # Keep training recipes auditable without making existing output directories
    # appear to contain complete training data in a fresh reproduction.
    for path in sorted((ROOT/'data/generated').glob('*/dataset_manifest.json')):
        if path.parent.name in {'workflow_confirmation_v1', 'workflow_confirmation_attributes_v1'}:
            continue
        add(path, Path('Temporal_Graph_Repair/data/recipe_manifests/generated')/path.parent.name/path.name)
    for path in sorted((ROOT/'data/edit_learning').glob('*/manifest.json')):
        add(path, Path('Temporal_Graph_Repair/data/recipe_manifests/edit_learning')/path.parent.name/path.name)
    for collection in ['workflow_confirmation_v1', 'workflow_confirmation_attributes_v1']:
        for path in sorted((ROOT/'data/generated'/collection).rglob('*')):
            if path.is_file():
                add(path)
    for folder in sorted((ROOT/'data/decoding').glob('confirm_generated_*')):
        for path in sorted(folder.glob('*.json')):
            add(path)
    factory = ROOT.parent/'RF_Dataset_Factory'
    for path in sorted((factory/'src/temporal_factory').glob('*.py')):
        add(path, Path('RF_Dataset_Factory')/path.relative_to(factory))
    for name in ['src/rfdf/__init__.py', 'src/rfdf/storage.py', 'pyproject.toml', 'tests/test_temporal_factory.py']:
        add(factory/name, Path('RF_Dataset_Factory')/name)
    destination.mkdir(parents=True)
    inventory = {}
    for name, path in sorted(files.items()):
        content = path.read_bytes()
        output = destination/name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        inventory[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    manifest = {
        'status': 'internal_incomplete_portability_check' if args.allow_incomplete else 'local_review_artifact',
        'protocol_hash': sha256(protocol_path), 'builder_hash': sha256(__file__),
        'selected_confirmation_status': evidence['status'],
        'source_availability': evidence['all_predeclared_sources'],
        'files': inventory,
        'scope': [
            'Private local review copy only; no external publication or license grant is performed.',
            'Full frozen generated test worlds, edit programs, model weights, calibration scores and analysis reports are included.',
            'Third-party operational datasets and their derived task graphs are omitted. Obtain the exact hashed inputs separately.',
            'Training manifests are recipes, not complete training caches. Regenerate before retraining.',
            'Only the bounded temporal factory component and its generic storage dependency are included.',
            'The historical factory directory name is retained for frozen source compatibility; no RF model, data or experiment is included.',
            'Absolute paths in the historical protocol record the original run. Use an explicitly declared path-only protocol copy for another installation.'
        ]}
    (destination/'REVIEW_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'destination': str(destination), 'files': len(inventory),
                      'bytes': sum(x['bytes'] for x in inventory.values()), 'status': manifest['status']}))


if __name__ == '__main__':
    main()
