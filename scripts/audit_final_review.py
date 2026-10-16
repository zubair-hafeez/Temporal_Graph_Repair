"""Check final report provenance and manuscript dependencies without rerunning models."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def main():
    checked = {}
    def read(name):
        path = ROOT/name
        checked[name] = sha256(path)
        return json.loads(path.read_text())
    protocol = read('docs/CONFIRMATION_PROTOCOL_V1.json')
    pins = (protocol['implementation_hashes'] | protocol['adapter_hashes'] |
            protocol['generated_confirmation']['script_hashes'] |
            protocol['target_local_adaptation']['script_hashes'])
    for name, digest in pins.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen implementation differs: '+name)
    specifications = [
        ('reports/confirmation_real_evidence_v1.json', 'complete_selected_confirmation'),
        ('reports/generated_confirmation_evidence_v1.json', 'complete_generated_confirmation_analysis'),
        ('reports/confirmation_candidate_evidence_v1.json', 'complete_candidate_analysis'),
        ('reports/calibration_sensitivity_evidence_v1.json', 'complete_post_freeze_calibration_analysis')]
    reports = {}
    for name, status in specifications:
        report = read(name)
        if report['status'] != status:
            raise ValueError('Incomplete final evidence: '+name)
        for source, digest in report['source_reports'].items():
            if sha256(ROOT/source) != digest:
                raise ValueError('Summarized source changed: '+source)
        reports[name] = report
    real = reports[specifications[0][0]]
    if real['coverage_counts'] != {'completed': 84}:
        raise ValueError('Unexpected selected confirmation coverage')
    if len(reports[specifications[3][0]]['trials']) != 9:
        raise ValueError('Incomplete calibration follow-up')
    for name, source, field in [
        ('paper/figures/confirmation_real_sources.json', specifications[0][0], 'evidence_sha256'),
        ('paper/tables/confirmation_real_sources.json', specifications[0][0], 'report_sha256'),
        ('paper/figures/generated_confirmation_sources.json', specifications[1][0], 'analysis_hash'),
        ('paper/tables/confirmation_candidates.sources.json', specifications[2][0], 'analysis_hash')]:
        if read(name)[field] != sha256(ROOT/source):
            raise ValueError('Figure/table uses stale analysis: '+name)
    manuscript = ROOT/'paper/main.tex'
    text = manuscript.read_text()
    if any(marker in text for marker in ['Internal research draft', 'The final study will test', 'TODO', 'FIXME']):
        raise ValueError('Unfinished manuscript marker')
    dependencies = re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', text)
    combined = text
    for name in dependencies:
        path = ROOT/'paper'/name
        if not path.exists():
            raise FileNotFoundError(path)
        checked[str(path.relative_to(ROOT))] = sha256(path)
        if path.suffix == '.tex':
            combined += '\n'+path.read_text()
    labels = re.findall(r'\\label\{([^}]+)\}', combined)
    if len(set(labels)) != len(labels):
        raise ValueError('Duplicate manuscript labels')
    missing = set(re.findall(r'\\ref\{([^}]+)\}', combined))-set(labels)
    if missing:
        raise ValueError('Unresolved references: '+repr(missing))
    cited = {key for group in re.findall(r'\\cite\{([^}]+)\}', combined) for key in group.split(',')}
    bibliography = set(re.findall(r'\\bibitem\{([^}]+)\}', text))
    if cited-bibliography:
        raise ValueError('Unresolved citations: '+repr(cited-bibliography))
    for name, status in [
        ('reports/portability_full_task_replay_v1.json', 'full_relocated_task_replay_passed'),
        ('reports/portability_raw_source_rebuild_v1.json', 'exact_public_source_split_reproduction')]:
        if read(name)['status'] != status:
            raise ValueError('Reproduction check failed')
    checked['paper/main.tex'] = sha256(manuscript)
    checked['paper/main.pdf'] = sha256(ROOT/'paper/main.pdf')
    build = read('reports/manuscript_build_v1.json')
    if build['status'] != 'manuscript_compiled' or build['pdf_sha256'] != checked['paper/main.pdf']:
        raise ValueError('PDF does not match its successful build record')
    for name, digest in build['inputs'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('PDF source changed after compilation: '+name)
    report = {'status': 'final_evidence_and_manuscript_dependencies_passed',
              'script_sha256': sha256(__file__), 'frozen_files_checked': len(pins),
              'original_selected_jobs': 84, 'post_freeze_sensitivity_jobs': 9,
              'sources_unexecuted': [k for k, v in real['all_predeclared_sources'].items()
                                     if not v['included_in_this_report']],
              'manuscript_citations': len(cited), 'dependencies_checked': checked,
              'limit': 'Provenance/completeness audit, not a peer-review judgment or acceptance guarantee.'}
    (ROOT/'reports/final_review_audit_v1.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'dependencies_checked'}))


if __name__ == '__main__':
    main()
