"""A concurrent caller must never receive an unfinished source-archive hash."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import tarfile
import threading

from tgrepair.provenance import snapshot_sources


def test_concurrent_source_snapshots_are_complete(tmp_path):
    root = tmp_path/'research'
    for folder in ['tgrepair', 'scripts', 'tests']:
        path = root/folder/'example.py'
        path.parent.mkdir(parents=True)
        path.write_text('example = '+repr('content '*10000)+'\n')
    factory = tmp_path/'RF_Dataset_Factory'
    for name in ['src/temporal_factory/core.py', 'src/rfdf/__init__.py',
                 'src/rfdf/storage.py', 'pyproject.toml']:
        path = factory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# bounded source fixture\n')
    barrier = threading.Barrier(8)
    def capture(_):
        barrier.wait(timeout=10)
        return snapshot_sources(root)
    with ThreadPoolExecutor(max_workers=8) as executor:
        reports = list(executor.map(capture, range(8)))
    assert len({report['source_set_hash'] for report in reports}) == 1
    for report in reports:
        archive_path = root/report['file']
        assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == report['sha256']
        with tarfile.open(archive_path, 'r:gz') as archive:
            hashes = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                      for member in archive.getmembers() if member.isfile()}
        assert hashes == report['files']
    assert not list((root/'data/source_snapshots').glob('*.tmp'))
