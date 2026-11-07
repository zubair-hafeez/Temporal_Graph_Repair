#!/usr/bin/env python3
"""Publish prepared research copies and full snapshot assets using an authenticated gh CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE = Path(__file__).resolve().parent
GH = BASE/'tools/gh'
ENV = dict(os.environ, GH_CONFIG_DIR=str(BASE/'.gh'))


def gh(*args, **kwargs):
    try:
        return subprocess.run([str(GH), *args], env=ENV, check=True, **kwargs)
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, flush=True)
        raise


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-release', action='store_true')
    parser.add_argument('projects', nargs='+', choices=('Temporal_Graph_Repair', 'Operational_Temporal_Graphs'))
    args = parser.parse_args()
    login = json.loads(gh('api', 'user', capture_output=True, text=True).stdout)['login']
    if login != 'zubair-hafeez':
        raise SystemExit('Authenticated account differs from selected owner: '+login)
    for project in args.projects:
        snapshot = BASE/'snapshots'/project
        prepared = json.loads((BASE/(project+'_prepared.json')).read_text())
        repo, tag = prepared['repository'], prepared['release_tag']
        source = Path(prepared['source_copy'])
        if args.inspect_release:
            gh('release', 'view', tag, '--repo', repo, '--json', 'id,apiUrl,isDraft,tagName,assets')
            continue
        view = subprocess.run([str(GH), 'repo', 'view', repo, '--json', 'nameWithOwner,isPrivate'],
                              env=ENV, capture_output=True, text=True)
        if view.returncode:
            gh('repo', 'create', repo, '--public', '--description',
               'Research code, manuscript, experiments and complete verified artifact snapshots')
        else:
            info = json.loads(view.stdout)
            if info['isPrivate']:
                raise SystemExit('Existing repository is private; refusing an automatic visibility change')
        git_env = dict(ENV, GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='credential.helper',
                       GIT_CONFIG_VALUE_0='!'+str(GH)+' auth git-credential')
        subprocess.run(['git', '-C', str(source), 'push', 'origin', 'HEAD:refs/heads/master'],
                       env=git_env, check=True)
        subprocess.run(['git', '-C', str(source), 'push', 'origin', '--tags'], env=git_env, check=True)
        release = subprocess.run([str(GH), 'release', 'view', tag, '--repo', repo],
                                 env=ENV, capture_output=True)
        if release.returncode:
            gh('release', 'create', tag, '--repo', repo, '--target', 'master', '--draft',
               '--title', 'Complete research snapshot', '--notes-file', str(snapshot/'RESTORE.md'))
        existing = json.loads(gh('release', 'view', tag, '--repo', repo, '--json', 'assets',
                                 capture_output=True, text=True).stdout)['assets']
        sizes = {a['name']: a['size'] for a in existing}
        for path in sorted(snapshot.iterdir()):
            if not path.is_file():
                continue
            if path.name in sizes:
                if sizes[path.name] != path.stat().st_size:
                    raise SystemExit('Existing release asset has a different size: '+path.name)
                print('Already uploaded:', path.name, flush=True)
                continue
            print('Uploading:', project, path.name, path.stat().st_size, flush=True)
            gh('release', 'upload', tag, str(path), '--repo', repo)
        details = json.loads(gh('release', 'view', tag, '--repo', repo, '--json',
                               'assets,apiUrl,isDraft,tagName', capture_output=True, text=True).stdout)
        expected = {p.name: p.stat().st_size for p in snapshot.iterdir() if p.is_file()}
        assets = {a['name']: a for a in details['assets']}
        for name, size in expected.items():
            if name not in assets or assets[name]['size'] != size:
                raise SystemExit('Remote asset is missing or size mismatch: '+name)
            remote_digest = assets[name].get('digest')
            local_digest = next((c['sha256'] for c in prepared['chunks'] if c['name'] == name), None)
            if local_digest is None:
                local_digest = digest(snapshot/name)
            if remote_digest != 'sha256:'+local_digest:
                raise SystemExit('Remote asset digest missing or mismatched: '+name)
        gh('release', 'edit', tag, '--repo', repo, '--draft=false')
        remote_head = subprocess.check_output(['git', '-C', str(source), 'ls-remote', 'origin',
                                              'refs/heads/master'], env=git_env, text=True).split()[0]
        local_head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if remote_head != local_head:
            raise SystemExit('Remote branch head does not match uploaded source')
        receipt = {'repository_url': 'https://github.com/'+repo,
                   'release_url': 'https://github.com/'+repo+'/releases/tag/'+tag,
                   'commit': local_head, 'remote_branch_verified': True,
                   'assets': [{'name': a['name'], 'size': a['size'], 'digest': a.get('digest')}
                              for a in details['assets']]}
        (BASE/(project+'_published.json')).write_text(json.dumps(receipt, indent=2)+'\n')
        print('PUBLISHED', receipt['repository_url'], receipt['release_url'], flush=True)


if __name__ == '__main__':
    main()
