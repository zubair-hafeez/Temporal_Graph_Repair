"""Validate historical archive contents and disclose transient checksum records."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def main():
    references = defaultdict(list)
    for path in sorted((ROOT/'reports').rglob('*.json')):
        report = json.loads(path.read_text())
        snapshot = report.get('source_snapshot', {})
        if snapshot.get('file'):
            references[snapshot['file']].append((path, snapshot))
    corrections, inventories = [], {}
    for name, rows in references.items():
        path = ROOT/name
        actual = sha256(path)
        with tarfile.open(path, 'r:gz') as archive:
            members = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                       for member in archive.getmembers() if member.isfile()}
        for source, snapshot in rows:
            digest = hashlib.sha256(json.dumps(snapshot['files'], sort_keys=True).encode()).hexdigest()
            if digest != snapshot['source_set_hash'] or members != snapshot['files']:
                raise ValueError('Source inventory mismatch: '+str(source))
            if actual != snapshot['sha256']:
                corrections.append({'report': str(source.relative_to(ROOT)), 'report_sha256': sha256(source),
                                    'archive': name, 'recorded_archive_sha256': snapshot['sha256'],
                                    'verified_archive_sha256': actual,
                                    'all_member_hashes_match_recorded_inventory': True})
        inventories[name] = {'sha256': actual, 'members': len(members), 'references': len(rows)}
    original = json.loads((ROOT/'reports/workflow_train_v4.json').read_text())['source_snapshot']
    factory = {}
    for name, digest in original['files'].items():
        if name.startswith('factory/'):
            path = ROOT.parent/'RF_Dataset_Factory'/name.removeprefix('factory/')
            if sha256(path) != digest:
                raise ValueError('Bounded factory changed since original training: '+name)
            factory[name] = digest
    result = {'status': 'all_archive_members_verified_with_historical_checksum_errata',
              'script_sha256': sha256(__file__), 'archives_checked': len(references),
              'references_checked': sum(map(len, references.values())), 'archives': inventories,
              'historical_checksum_errata': corrections, 'unchanged_factory_files': factory,
              'explanation': 'The historical writer exposed a destination before finishing its compressed archive. Concurrent callers could record an empty or partial byte checksum. All archived member bytes match every declared per-file inventory; original report fields are preserved and corrected archive hashes are listed here. Model, calibration and dataset hashes are separate and unaffected.',
              'scope': 'No historical result, model, dataset, source member or original report is rewritten.'}
    (ROOT/'reports/source_snapshot_audit_v1.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'archives': len(references),
                      'references': result['references_checked'], 'errata': len(corrections)}))


if __name__ == '__main__':
    main()
