"""Build the latest manuscript without overwriting historical build evidence."""
import json
import os
import re
import subprocess

from .run import HERE, ROOT, write
from tgrepair.data import sha256


def main():
    paper = ROOT / 'revision/paper'
    compiler = '/mnt/zubair/Operational_Temporal_Graphs/paper/tools/tectonic'
    sources = list(paper.glob('*.tex')) + list(HERE.glob('paper_table_*.tex'))
    tex = '\n'.join(path.read_text() for path in sources)
    assert '\\pending{' not in tex
    for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex):
        sources.append((paper / name).resolve())
    inputs = {str(path.resolve().relative_to(ROOT.resolve())): sha256(path) for path in sources}
    env = os.environ.copy(); env['XDG_CACHE_HOME'] = str(ROOT / 'cache/tex-cache')
    command = [compiler, '--only-cached', '--keep-logs', 'main.tex']
    result = subprocess.run(command, cwd=paper, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end='')
    if result.returncode: raise RuntimeError('LaTeX build failed')
    for path, digest in inputs.items(): assert sha256(ROOT / path) == digest
    log = (paper / 'main.log').read_text()
    unresolved = re.findall(r'.*(?:undefined references|undefined citations|Citation .* undefined|Reference .* undefined|multiply defined).*', log)
    assert not unresolved, unresolved
    info = subprocess.check_output(['pdfinfo', str(paper / 'main.pdf')], text=True)
    pages = int(re.search(r'^Pages:\s*(\d+)', info, re.M).group(1))
    record = {'status': 'built_for_review', 'pages': pages, 'page_limit': 15,
              'inputs': inputs, 'pdf_sha256': sha256(paper / 'main.pdf'),
              'log_sha256': sha256(paper / 'main.log'), 'command': command,
              'overfull': re.findall(r'.*Overfull.*', log), 'script_sha256': sha256(__file__),
              'analysis_sha256': sha256(HERE / 'analysis.json')}
    write(HERE / 'build.json', record)
    print(json.dumps({'pages': pages, 'overfull': record['overfull']}))
    assert pages <= 15, 'Total page limit exceeded'
    assert not record['overfull'], 'Inspect layout overflow'


if __name__ == '__main__':
    main()
