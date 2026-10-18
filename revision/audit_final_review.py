"""Check completed evidence, claim receipts and a recorded manuscript review.

This gate cannot decide novelty or acceptance. It rejects incomplete evidence,
stale numerical claims, pending manuscript text and an unreviewed PDF.
"""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.confirmation import verify_protocol
from revision.native_protocol import verify_protocol as verify_native
from tgrepair.data import sha256


def main():
    protocol = verify_protocol()
    verify_native('bpi2017', 0, .3, 30, 2048)
    records = {
        'revision/reports/coupled_confirmation_audit_v1.json': 'all_coupled_confirmation_predictions_and_edit_replays_verified',
        'revision/reports/native_confirmation_audit_v1.json': 'all_native_confirmation_predictions_verified',
        'revision/reports/coupled_summary_v1.json': 'complete_audited_coupled_summary',
        'revision/reports/revision_final_build_v1.json': 'final_revision_compiled',
        'revision/reports/manuscript_review_v1.json': 'manuscript_and_visual_review_completed',
    }
    checked, data = {}, {}
    for name, status in records.items():
        value = json.loads((ROOT / name).read_text())
        if value['status'] != status:
            raise AssertionError('Incomplete required review: ' + name)
        checked[name] = sha256(ROOT / name)
        data[name] = value
    build = data['revision/reports/revision_final_build_v1.json']
    if not 1 <= build['page_count'] <= 15 or build['overfull_warnings']:
        raise AssertionError('Page-count or layout gate failed')
    for name, digest in build['inputs'].items():
        if sha256(ROOT / name) != digest:
            raise AssertionError('Manuscript source changed since build')
        checked[name] = digest
    pdf = 'revision/paper/main.pdf'
    if sha256(ROOT / pdf) != build['pdf_sha256']:
        raise AssertionError('PDF differs from final build')
    checked[pdf] = sha256(ROOT / pdf)
    review = data['revision/reports/manuscript_review_v1.json']
    if review['pdf_sha256'] != build['pdf_sha256']:
        raise AssertionError('Visual review concerns a different PDF')
    required = ['all_pages_visually_inspected', 'negative_results_retained',
                'claims_match_available_evidence', 'public_data_and_fault_scope_explicit',
                'verification_budget_counts_calibration', 'no_acceptance_guarantee',
                'original_study_preserved', 'no_pending_research_text']
    if any(review['checks'].get(name) is not True for name in required):
        raise AssertionError('Recorded scientific/visual review is incomplete')
    tex = '\n'.join(p.read_text() for p in (ROOT / 'revision/paper').glob('*.tex'))
    if '\\pending{' in tex or 'Working revision' in tex:
        raise AssertionError('Unfinished manuscript text')
    keys = set(re.findall(r'\\bibitem\{([^}]+)\}', tex))
    citations = {key.strip() for group in re.findall(r'\\cite\{([^}]+)\}', tex) for key in group.split(',')}
    if not citations <= keys:
        raise AssertionError('Missing bibliography entries')

    claims_path = ROOT / 'revision/reports/numerical_claims_v1.json'
    claims = json.loads(claims_path.read_text())
    if not claims['claims']:
        raise AssertionError('Provide explicit numeric claim receipts')
    for claim in claims['claims']:
        path = ROOT / claim['source']
        if sha256(path) != claim['source_sha256']:
            raise AssertionError('Claim source changed: ' + claim['id'])
        value = json.loads(path.read_text())
        for key in claim['field_path']:
            value = value[key]
        if isinstance(value, (int, float)):
            if abs(value - claim['value']) > 1e-12:
                raise AssertionError('Claim numeric value differs: ' + claim['id'])
        elif value != claim['value']:
            raise AssertionError('Claim value differs: ' + claim['id'])
        if claim['manuscript_literal'] not in tex:
            raise AssertionError('Claim text absent from manuscript: ' + claim['id'])
        checked[claim['source']] = claim['source_sha256']
    checked[str(claims_path.relative_to(ROOT))] = sha256(claims_path)
    for name, value in data.items():
        if 'inputs' in value and name.endswith('audit_v1.json'):
            for source, digest in value['inputs'].items():
                if sha256(ROOT / source) != digest:
                    raise AssertionError('Audited evidence changed')
                checked[source] = digest
    for group in ['implementation', 'inputs', 'models']:
        checked.update(protocol[group])
    original_protocol = 'docs/CONFIRMATION_PROTOCOL_V1.json'
    expected = 'a3e12b092cc7670f8b4d2c7190818eb2c04ea88e7ae4a2afb20bdffac691c25d'
    if sha256(ROOT / original_protocol) != expected:
        raise AssertionError('Original frozen protocol changed')
    checked[original_protocol] = expected
    result = {'status': 'ready_for_author_review', 'page_count': build['page_count'],
              'scope': 'Internal evidence, numeric receipt and recorded visual review. Does not establish external peer review, independent natural-error truth, or acceptance.',
              'checked_files': checked, 'script_sha256': sha256(__file__),
              'numerical_claims': len(claims['claims']), 'coupled_jobs': len(protocol['jobs']),
              'native_conditions': len(data['revision/reports/native_confirmation_audit_v1.json']['jobs'])}
    destination = ROOT / 'revision/reports/final_revision_review_v1.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checked_files'}))


if __name__ == '__main__':
    main()
