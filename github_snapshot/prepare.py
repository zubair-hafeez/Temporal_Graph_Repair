#!/usr/bin/env python3
"""Prepare isolated Git copies and complete, verified GitHub Release snapshots."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
PROJECTS = ('Temporal_Graph_Repair', 'Operational_Temporal_Graphs')
CHUNK = 1_900_000_000
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def inventory(source):
    entries = []
    for directory, dirs, files in os.walk(source, followlinks=False):
        dirs.sort()
        files.sort()
        for name in dirs + files:
            path = Path(directory) / name
            s = path.lstat()
            row = {'path': path.relative_to(source).as_posix(),
                   'mode': stat.S_IMODE(s.st_mode), 'mtime_ns': s.st_mtime_ns}
            if path.is_symlink():
                row.update(type='symlink', target=os.readlink(path))
            elif path.is_dir():
                row.update(type='directory')
            elif path.is_file():
                row.update(type='file', size=s.st_size, sha256=digest(path))
                if s.st_size <= 8 * 1024**2 and '.git' not in path.relative_to(source).parts:
                    data = path.read_bytes()
                    if b'\0' not in data and SECRET.search(data):
                        raise RuntimeError('Possible credential; publishing stopped: ' + row['path'])
                if path.stat().st_size != s.st_size or path.stat().st_mtime_ns != s.st_mtime_ns:
                    raise RuntimeError('File changed during hashing: ' + str(path))
            else:
                raise RuntimeError('Unsupported filesystem entry: ' + str(path))
            entries.append(row)
    return entries


def scan_history(source):
    objects = subprocess.check_output(['git', '-C', str(source), 'rev-list', '--objects', '--all'])
    ids = [line.split(b' ', 1)[0] for line in objects.splitlines()]
    process = subprocess.Popen(['git', '-C', str(source), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for oid in ids:
            process.stdin.write(oid + b'\n')
            process.stdin.flush()
            header = process.stdout.readline().split()
            size = int(header[2])
            data = process.stdout.read(size)
            process.stdout.read(1)
            if header[1] == b'blob' and b'\0' not in data and SECRET.search(data):
                raise RuntimeError('Possible credential in Git history object ' + oid.decode())
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError('Git history credential scan failed')


def archive(source, output):
    tar = subprocess.Popen(['tar', '--format=posix', '--hard-dereference', '-C', str(source.parent), '-cf', '-', source.name],
                           stdout=subprocess.PIPE)
    gzip = subprocess.Popen(['pigz', '-p', '4', '-1', '-n'], stdin=tar.stdout, stdout=subprocess.PIPE)
    tar.stdout.close()
    chunks = []
    out = None
    try:
        while True:
            data = gzip.stdout.read(8 * 1024**2)
            if not data:
                break
            while data:
                if out is None:
                    path = output / ('snapshot.tar.gz.part%03d' % len(chunks))
                    if path.exists():
                        raise RuntimeError('Refusing to overwrite ' + str(path))
                    out = path.open('xb')
                    h = hashlib.sha256()
                    size = 0
                block, data = data[:CHUNK-size], data[CHUNK-size:]
                out.write(block)
                h.update(block)
                size += len(block)
                if size == CHUNK:
                    out.close()
                    chunks.append({'name': path.name, 'size': size, 'sha256': h.hexdigest()})
                    print('  created', path.name, size, flush=True)
                    out = None
        if out is not None:
            out.close()
            chunks.append({'name': path.name, 'size': size, 'sha256': h.hexdigest()})
        if gzip.wait() or tar.wait():
            raise RuntimeError('Archive compression failed')
    finally:
        if out is not None and not out.closed:
            out.close()
    return chunks


class Joined:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.file = None

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


def verify(project, output, entries, chunks):
    expected = {project + '/' + e['path']: e for e in entries}
    seen = set()
    stream = Joined([output/c['name'] for c in chunks])
    with tarfile.open(fileobj=stream, mode='r|gz') as archive_file:
        for member in archive_file:
            name = member.name.rstrip('/')
            if name == project:
                continue
            if name not in expected or name in seen:
                raise RuntimeError('Unexpected or duplicate archive member: ' + name)
            row = expected[name]
            seen.add(name)
            if stat.S_IMODE(member.mode) != row['mode']:
                raise RuntimeError('Mode mismatch: ' + name)
            if row['type'] == 'file':
                if not member.isfile() or member.size != row['size']:
                    raise RuntimeError('File size/type mismatch: ' + name)
                h = hashlib.sha256()
                with archive_file.extractfile(member) as f:
                    for block in iter(lambda: f.read(8 * 1024**2), b''):
                        h.update(block)
                if h.hexdigest() != row['sha256']:
                    raise RuntimeError('Archived content mismatch: ' + name)
            elif row['type'] == 'symlink':
                if not member.issym() or member.linkname != row['target']:
                    raise RuntimeError('Symlink mismatch: ' + name)
            elif not member.isdir():
                raise RuntimeError('Directory mismatch: ' + name)
    if seen != set(expected):
        raise RuntimeError('Missing archive members: ' + str(set(expected)-seen))
    for c in chunks:
        if digest(output/c['name']) != c['sha256']:
            raise RuntimeError('Chunk checksum mismatch: ' + c['name'])
    return {'archive_member_checks': len(seen), 'file_content_sha256': 'passed',
            'symlinks_and_permissions': 'passed', 'chunk_checksums': 'passed'}


def make_source_copy(source, output, release_tag):
    dest = BASE / 'repositories' / source.name
    dest.parent.mkdir(exist_ok=True)
    run(['git', 'clone', '--no-hardlinks', str(source), str(dest)], stdout=subprocess.DEVNULL)
    raw = subprocess.check_output(['git', '-C', str(source), 'ls-files', '--cached', '--others',
                                   '--exclude-standard', '-z'])
    paths = sorted(set(os.fsdecode(raw).split('\0')) - {''})
    for relative in paths:
        src, dst = source/relative, dest/relative
        if not src.exists() and not src.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            continue
        if not src.is_symlink() and src.stat().st_size >= 100 * 1024**2:
            raise RuntimeError('Oversized nonignored Git file: ' + str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            dst.unlink()
        if src.is_symlink():
            dst.symlink_to(os.readlink(src))
        else:
            shutil.copy2(src, dst)
    docs = dest/'github_snapshot'
    docs.mkdir(exist_ok=True)
    for name in ('inventory.json', 'snapshot.json', 'SHA256SUMS', 'VERIFICATION_REPORT.md', 'RESTORE.md'):
        shutil.copy2(output/name, docs/name)
    shutil.copy2(BASE/'restore.py', docs/'restore.py')
    shutil.copy2(BASE/'prepare.py', docs/'prepare.py')
    shutil.copy2(BASE/'publish.py', docs/'publish.py')
    readme = dest/'README.md'
    existing = readme.read_text()
    notice = ('> **Complete research snapshot:** Code and manuscript files are browsable here. '
              'Datasets, checkpoints, caches, ignored files and original local Git metadata are included '
              'in the split archives attached to [release ' + release_tag + '](https://github.com/zubair-hafeez/'
              + source.name + '/releases/tag/' + release_tag + '). '
              'See [restore instructions](github_snapshot/RESTORE.md) and '
              '[verification report](github_snapshot/VERIFICATION_REPORT.md). '
              'A normal Git clone does not download those archives.\n\n')
    readme.write_text(notice + existing)
    run(['git', '-C', str(dest), 'remote', 'set-url', 'origin',
         'https://github.com/zubair-hafeez/' + source.name + '.git'])
    run(['git', '-C', str(dest), 'add', '-A'])
    run(['git', '-C', str(dest), '-c', 'user.name=Codex Backup',
         '-c', 'user.email=codex-backup@users.noreply.github.com', 'commit', '-m',
         'Preserve current research, manuscript and verified full-snapshot manifest'],
        stdout=subprocess.DEVNULL)
    return str(dest)


def prepare(project):
    source = ROOT/project
    output = BASE/'snapshots'/project
    output.mkdir(parents=True, exist_ok=False)
    tag = 'research-snapshot-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    print(project + ': scanning history and hashing all files', flush=True)
    scan_history(source)
    entries = inventory(source)
    (output/'inventory.json').write_text(json.dumps(entries, indent=2) + '\n')
    print(project + ': packaging complete folder, including .git and ignored files', flush=True)
    chunks = archive(source, output)
    print(project + ': verifying every archived file against its original SHA-256', flush=True)
    verification = verify(project, output, entries, chunks)
    # Check the source did not change after it was hashed and archived.
    for e in entries:
        s = (source/e['path']).lstat()
        if s.st_mtime_ns != e['mtime_ns'] or (e['type'] == 'file' and s.st_size != e['size']):
            raise RuntimeError('Source changed during snapshot: ' + e['path'])
    summary = {'project': project, 'repository': 'zubair-hafeez/'+project, 'release_tag': tag,
               'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'files': sum(e['type']=='file' for e in entries),
               'symlinks': sum(e['type']=='symlink' for e in entries),
               'original_file_bytes': sum(e.get('size', 0) for e in entries),
               'archive_bytes': sum(c['size'] for c in chunks), 'chunks': chunks,
               'includes_ignored_files': True, 'includes_original_git_metadata': True,
               'verification': verification}
    (output/'snapshot.json').write_text(json.dumps(summary, indent=2) + '\n')
    sums = [(c['sha256'], c['name']) for c in chunks]
    sums += [(digest(output/'inventory.json'), 'inventory.json'),
             (digest(output/'snapshot.json'), 'snapshot.json')]
    (output/'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for h,n in sums))
    (output/'RESTORE.md').write_text(
        '# Restore the complete research snapshot\n\n'
        'A Git clone contains code, reports and papers; it does not contain the complete data/models.\n\n'
        'Download every asset from release `' + tag + '` into one folder, including '
        '`restore.py`, `inventory.json`, `snapshot.json`, `SHA256SUMS` and all '
        '`snapshot.tar.gz.partNNN` files. Then run:\n\n'
        '```bash\npython3 restore.py --snapshot-dir . --destination /path/to/empty-restore-folder\n```\n\n'
        'The destination must not already exist. The script checks checksums, safely extracts the '
        'archive, and verifies every restored file and symlink. The restored project includes '
        'the original `.git` directory, including local history and the original uncommitted state.\n\n'
        'The snapshot includes all files present in the original project, including ignored data, '
        'models and caches. Third-party datasets and code retain their original terms; this backup '
        'does not grant a new license. Preserve attribution and consult recorded source provenance '
        'before redistributing or reusing third-party material. Files referenced outside the project '
        'are not included. A referenced Temporal_Walk_Counterfactual_Audits directory was absent '
        'at preparation time; this snapshot cannot recover that missing directory.\n')
    (output/'VERIFICATION_REPORT.md').write_text(
        '# Snapshot verification\n\n'
        '- All current project files, hidden files, ignored files and original Git metadata included.\n'
        '- Every archived regular file SHA-256 verified against the source inventory.\n'
        '- Archive membership, symbolic-link targets and POSIX permission bits checked.\n'
        '- Every split archive checksum independently checked.\n'
        '- Source sizes and modification times checked for changes during preparation.\n'
        '- Working files and reachable Git history scanned for common credential signatures; '
        'this is a bounded automated scan, not a guarantee of absence.\n'
        '- Original research folders were not modified.\n'
        '- No experiments were rerun and no scientific claims were independently revalidated.\n'
        '- Remote upload and remote verification are recorded separately in the publisher receipt.\n'
        '- Commit author Codex Backup identifies the automated packaging operation.\n')
    shutil.copy2(BASE/'restore.py', output/'restore.py')
    summary['source_copy'] = make_source_copy(source, output, tag)
    (BASE/(project+'_prepared.json')).write_text(json.dumps(summary, indent=2)+'\n')
    print(project + ': READY, compressed bytes ' + str(summary['archive_bytes']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('projects', nargs='*', choices=PROJECTS)
    args = parser.parse_args()
    for project in args.projects or PROJECTS:
        prepare(project)
