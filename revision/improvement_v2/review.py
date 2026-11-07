"""Final checks for the updated manuscript; historical review reports stay intact."""
import json
import re

from revision.confirmation import verify_protocol as verify_previous
from tgrepair.data import sha256
from .run import HERE, ROOT, verify, write


def main():
    verify(); verify_previous()
    analysis = json.loads((HERE / 'analysis.json').read_text())
    assert analysis['conditions'] == 156 and analysis['models_with_replayed_calibration'] == 60
    assert sha256(HERE / 'analyze.py') == analysis['analysis_code_sha256']
    for name, digest in analysis['checked_files'].items(): assert sha256(HERE / name) == digest
    build = json.loads((HERE / 'build.json').read_text())
    assert build['pages'] <= 15 and not build['overfull']
    assert build['analysis_sha256'] == sha256(HERE / 'analysis.json')
    for path, digest in build['inputs'].items(): assert sha256(ROOT / path) == digest
    assert sha256(ROOT / 'revision/paper/main.pdf') == build['pdf_sha256']
    visual = json.loads((HERE / 'visual_review.json').read_text())
    assert visual['pdf_sha256'] == build['pdf_sha256']
    assert visual['inspected_pages'] == list(range(1, build['pages'] + 1))
    assert visual['no_clipped_or_overlapping_content']
    presentation = json.loads((HERE / 'presentation.json').read_text())
    assert presentation['analysis_sha256'] == sha256(HERE / 'analysis.json')
    assert sha256(HERE / 'present.py') == presentation['script_sha256']
    for name, digest in presentation['outputs'].items(): assert sha256(HERE / name) == digest
    tex = '\n'.join(path.read_text() for path in (ROOT / 'revision/paper').glob('*.tex'))
    tex += '\n' + '\n'.join(path.read_text() for path in HERE.glob('paper_table_*.tex'))
    keys = set(re.findall(r'\\bibitem\{([^}]+)\}', tex))
    cited = {key.strip() for group in re.findall(r'\\cite\{([^}]+)\}', tex) for key in group.split(',')}
    assert cited <= keys
    claims = json.loads((HERE / 'claims.json').read_text())
    for claim in claims:
        path = HERE / claim['source']
        assert sha256(path) == claim['source_sha256']
        value = json.loads(path.read_text())
        for key in claim['field_path']: value = value[key]
        assert abs(value - claim['value']) < 1e-12
        assert claim['literal'] in tex
    original = ROOT / 'docs/CONFIRMATION_PROTOCOL_V1.json'
    assert sha256(original) == 'a3e12b092cc7670f8b4d2c7190818eb2c04ea88e7ae4a2afb20bdffac691c25d'
    write(HERE / 'review.json', {
        'status': 'updated_paper_ready_for_author_review', 'pages': build['pages'],
        'new_fitted_models': 60, 'new_repair_conditions': 156, 'claims_checked': len(claims),
        'pdf_sha256': build['pdf_sha256'], 'protocol_sha256': sha256(HERE / 'protocol.json'),
        'analysis_sha256': sha256(HERE / 'analysis.json'), 'build_sha256': sha256(HERE / 'build.json'),
        'visual_review_sha256': sha256(HERE / 'visual_review.json'), 'script_sha256': sha256(__file__),
        'delivery_files': {name: sha256(HERE / name) for name in ['RESULTS.txt', 'per_trial.csv', 'claims.json',
                           'REPRODUCE.txt', 'generation_benchmark.json', 'presentation.json']},
        'previous_frozen_implementation_inputs_and_models_unchanged': True,
        'scope': 'Exploratory reused-cohort follow-up, complete fixed-grid outcomes retained; no new independent natural-error or human-time evidence.'})
    print(json.dumps({'status': 'updated_paper_ready_for_author_review', 'pages': build['pages'], 'conditions': 156}))


if __name__ == '__main__':
    main()
