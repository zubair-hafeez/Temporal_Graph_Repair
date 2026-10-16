"""Schema/ground-truth compatibility audit; no model performance is computed."""
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def main():
    directory = ROOT / 'revision/data/natural_feasibility'
    paths = {kind: directory / f'arnetminer_{kind}.json' for kind in ['signatures', 'papers', 'clusters']}
    data = {kind: json.loads(path.read_text()) for kind, path in paths.items()}
    truth = {}
    for cid, cluster in data['clusters'].items():
        for sid in cluster['signature_ids']:
            sid = str(sid)
            if sid in truth:
                raise ValueError('Signature appears in multiple curated clusters')
            truth[sid] = str(cid)
    if set(truth) != set(data['signatures']):
        raise ValueError('Curated reference does not cover signatures exactly')
    profiles = defaultdict(set)
    identities = defaultdict(set)
    missing_profile_identities = set()
    missing_profiles = 0
    years = []
    for sid, record in data['signatures'].items():
        cluster = truth[sid]
        if record['author_id'] is None:
            missing_profiles += 1
            missing_profile_identities.add(cluster)
            continue
        profile = str(record['author_id'])
        profiles[profile].add(cluster)
        identities[cluster].add(profile)
        paper = data['papers'].get(str(record['paper_id']), {})
        if isinstance(paper.get('year'), int):
            years.append(paper['year'])
    impure = {p for p, clusters in profiles.items() if len(clusters) > 1}
    touches_impure = set().union(*(profiles[p] for p in impure))
    excluded = touches_impure | missing_profile_identities
    pure = {c: ps for c, ps in identities.items() if c not in excluded}
    counts = {'signatures': len(data['signatures']), 'curated_identity_clusters': len(data['clusters']),
              'observed_author_profiles': len(profiles), 'missing_profile_signatures': missing_profiles,
              'impure_observed_profiles': len(impure),
              'curated_identities_touching_impure_profiles': len(touches_impure),
              'curated_identities_touching_missing_profiles': len(missing_profile_identities),
              'remaining_complete_pure_identities': len(pure),
              'remaining_fragmented_identities': sum(len(ps) > 1 for ps in pure.values()),
              'remaining_true_profile_pairs': sum(len(ps) * (len(ps) - 1) // 2 for ps in pure.values()),
              'year_records': len(years), 'minimum_year': min(years), 'maximum_year': max(years)}
    report = {'status': 'completed_source_compatibility_audit_no_model_evaluation',
              'source': 'S2AND ArnetMiner',
              'sha256': {kind: sha256(path) for kind, path in paths.items()},
              'primary_reference': 'https://arxiv.org/abs/2103.07534',
              'source_distribution': 'https://github.com/allenai/S2AND/tree/s2and_paper',
              'counts': counts,
              'finding': 'Observed production author profiles frequently combine multiple curated identities. The current repair contract maps each handle to one true identity and has no split/new-entity operation, so these records do not directly instantiate the claimed task.',
              'decision': 'Do not present this dataset as natural business-error validation. A pure-profile-only analysis would exclude most curated identities and is not representative. Retain this audit and keep natural operational-error validation as an explicit limitation unless another compatible public source is verified.',
              'user_constraint': 'Public data only, explicitly confirmed by the user during revision.',
              'performance_selection': False,
              'individual_names_or_emails_released_by_audit': False}
    out = ROOT / 'revision/reports/natural_source_compatibility_v1.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
