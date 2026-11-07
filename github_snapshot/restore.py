#!/usr/bin/env python3
"""Verify and restore a complete split research snapshot without overwriting files."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


class Joined:
    def __init__(self, paths):
        self.paths, self.file = iter(paths), None

    def read(self, size):
        result = bytearray()
        while len(result) < size:
            if self.file is None:
                try:
                    self.file = next(self.paths).open('rb')
                except StopIteration:
                    break
            data = self.file.read(size-len(result))
            if data:
                result.extend(data)
            else:
                self.file.close()
                self.file = None
        return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    folder, dest = args.snapshot_dir.resolve(), args.destination.absolute()
    if dest.exists() or dest.is_symlink():
        raise SystemExit('Destination must not already exist: ' + str(dest))
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        h, name = line.split('  ', 1)
        if Path(name).name != name or digest(folder/name) != h:
            raise SystemExit('Checksum failed: ' + name)
    summary = json.loads((folder/'snapshot.json').read_text())
    inventory = json.loads((folder/'inventory.json').read_text())
    project = summary['project']
    if Path(project).name != project:
        raise SystemExit('Invalid project name')
    expected = {project+'/'+e['path']: e for e in inventory}
    chunks = [folder/c['name'] for c in summary['chunks']]
    dest.mkdir(parents=True)
    seen, links = set(), []
    with tarfile.open(fileobj=Joined(chunks), mode='r|gz') as archive:
        for member in archive:
            name = member.name.rstrip('/')
            parts = PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts:
                raise SystemExit('Unsafe archive path: ' + name)
            target = dest.joinpath(*parts.parts)
            if name == project:
                target.mkdir(exist_ok=True)
                continue
            if name not in expected or name in seen:
                raise SystemExit('Unexpected archive member: ' + name)
            row = expected[name]
            seen.add(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if row['type'] == 'file' and member.isfile():
                h = hashlib.sha256()
                with archive.extractfile(member) as src, target.open('xb') as out:
                    for block in iter(lambda: src.read(8 * 1024**2), b''):
                        out.write(block)
                        h.update(block)
                if target.stat().st_size != row['size'] or h.hexdigest() != row['sha256']:
                    raise SystemExit('Restored file content mismatch: ' + name)
                target.chmod(row['mode'])
                os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))
            elif row['type'] == 'directory' and member.isdir():
                target.mkdir(exist_ok=True)
            elif row['type'] == 'symlink' and member.issym() and member.linkname == row['target']:
                links.append((target, row))
            else:
                raise SystemExit('Archive type mismatch: ' + name)
    if seen != set(expected):
        raise SystemExit('Archive is missing inventoried members')
    # Create links after writing regular files so archive links cannot redirect extraction.
    for target, row in links:
        target.symlink_to(row['target'])
        if os.readlink(target) != row['target']:
            raise SystemExit('Symlink verification failed')
        os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']), follow_symlinks=False)
    for row in sorted(inventory, key=lambda e: len(PurePosixPath(e['path']).parts), reverse=True):
        if row['type'] == 'directory':
            path = dest/project/row['path']
            path.chmod(row['mode'])
            os.utime(path, ns=(row['mtime_ns'], row['mtime_ns']))
    print('Restored and verified', len(seen), 'entries to', dest/project)


if __name__ == '__main__':
    main()
