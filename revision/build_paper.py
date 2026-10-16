"""Build the isolated revision and enforce its total 15-page limit."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--compiler', default='/mnt/zubair/Operational_Temporal_Graphs/paper/tools/tectonic')
    args = parser.parse_args()
    paper = ROOT / 'revision/paper'
    sources = list(paper.glob('*.tex'))
    contents = '\n'.join(p.read_text() for p in sources)
    if not args.draft and ('\\pending{' in contents or 'Working revision' in contents):
        raise ValueError('Final build cannot contain pending research claims')
    for source in list(sources):
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source.read_text()):
            sources.append((paper / name).resolve())
    inputs = {str(p.resolve().relative_to(ROOT.resolve())): sha256(p) for p in sources}
    env = os.environ.copy()
    env['XDG_CACHE_HOME'] = str(ROOT / 'cache/tex-cache')
    command = [args.compiler, '--only-cached', '--keep-logs', 'main.tex']
    result = subprocess.run(command, cwd=paper, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    for path, digest in inputs.items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Paper dependency changed during compilation: ' + path)
    log = (paper / 'main.log').read_text(errors='replace')
    unresolved = re.findall(r'.*(?:undefined references|undefined citations|Citation .* undefined|Reference .* undefined|multiply defined).*', log)
    if unresolved:
        raise ValueError('Unresolved references: ' + repr(unresolved))
    info = subprocess.check_output(['pdfinfo', str(paper / 'main.pdf')], text=True)
    pages = int(re.search(r'^Pages:\s*(\d+)', info, re.M).group(1))
    if pages > 15:
        raise ValueError(f'User limit exceeded: {pages} total pages')
    report = {'status': 'development_draft_compiled' if args.draft else 'final_revision_compiled',
              'page_count': pages, 'page_limit_includes_references_and_appendices': 15,
              'compiler_version': subprocess.check_output([args.compiler, '--version'], text=True).strip(),
              'command': command, 'inputs': inputs, 'script_sha256': sha256(__file__),
              'pdf_sha256': sha256(paper / 'main.pdf'), 'log_sha256': sha256(paper / 'main.log'),
              'overfull_warnings': re.findall(r'.*Overfull.*', log),
              'scope': 'Build and page-count verification. Scientific completeness, numerical claims and visual layout require separate review.'}
    name = 'revision_draft_build_v1.json' if args.draft else 'revision_final_build_v1.json'
    (ROOT / 'revision/reports' / name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'pages': pages, 'overfull': len(report['overfull_warnings'])}))


if __name__ == '__main__':
    main()
