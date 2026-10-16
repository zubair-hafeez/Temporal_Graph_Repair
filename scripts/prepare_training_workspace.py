"""Copy code into a new empty workspace for independent recipe execution."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    destination = Path(args.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    files = {}
    for folder in ['tgrepair', 'scripts', 'tests']:
        for source in (ROOT/folder).glob('*.py'):
            files[Path('Temporal_Graph_Repair')/source.relative_to(ROOT)] = source
    files[Path('Temporal_Graph_Repair/requirements.txt')] = ROOT/'requirements.txt'
    factory = ROOT.parent/'RF_Dataset_Factory'
    for source in (factory/'src/temporal_factory').glob('*.py'):
        files[Path('RF_Dataset_Factory')/source.relative_to(factory)] = source
    for name in ['src/rfdf/__init__.py', 'src/rfdf/storage.py', 'pyproject.toml', 'tests/test_temporal_factory.py']:
        files[Path('RF_Dataset_Factory')/name] = factory/name
    inventory = {}
    destination.mkdir(parents=True)
    for relative, source in files.items():
        content = source.read_bytes()
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        inventory[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    for folder in ['reports', 'data', 'checkpoints']:
        (destination/'Temporal_Graph_Repair'/folder).mkdir()
    manifest = {'status': 'empty_training_workspace', 'copied_source_sha256': inventory,
                'scope': 'No model weights, training examples, reports or evaluation references are copied.'}
    (destination/'TRAINING_WORKSPACE.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'workspace': str(destination/'Temporal_Graph_Repair'), 'copied_files': len(inventory)}))


if __name__ == '__main__':
    main()
