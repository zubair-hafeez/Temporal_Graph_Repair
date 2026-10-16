"""Compile the manuscript and record exactly which sources produced the PDF."""
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
    parser.add_argument('--compiler', default='tectonic')
    parser.add_argument('--cache')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    manuscript = ROOT/'paper/main.tex'
    dependencies = re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', manuscript.read_text())
    paths = [manuscript]+[ROOT/'paper'/name for name in dependencies]
    inputs = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    environment = os.environ.copy()
    if args.cache:
        environment['XDG_CACHE_HOME'] = str(Path(args.cache).resolve())
    command = [args.compiler, '--keep-logs']
    if args.offline:
        command.append('--only-cached')
    command.append('main.tex')
    result = subprocess.run(command, cwd=ROOT/'paper', env=environment,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='')
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    for path, digest in inputs.items():
        if sha256(ROOT/path) != digest:
            raise ValueError('Manuscript input changed during compilation: '+path)
    log = (ROOT/'paper/main.log').read_text(errors='replace')
    unresolved = re.findall(r'.*(?:undefined references|undefined citations|Citation .* undefined|Reference .* undefined|multiply defined).*', log)
    if unresolved:
        raise ValueError('Unresolved TeX references: '+repr(unresolved))
    report = {'status': 'manuscript_compiled', 'command': command,
              'compiler_version': subprocess.check_output([args.compiler, '--version'], text=True).strip(),
              'script_sha256': sha256(__file__), 'inputs': inputs,
              'pdf_sha256': sha256(ROOT/'paper/main.pdf'), 'log_sha256': sha256(ROOT/'paper/main.log'),
              'scope': 'Build correspondence only; page appearance is reviewed separately.'}
    (ROOT/'reports/manuscript_build_v1.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
