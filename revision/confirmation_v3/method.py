"""Fixed-policy comparison; never-merge changes only merge eligibility."""
from revision.improvement_v2.method import decode


def apply_policy(observation, scores, thresholds, root_type, policy):
    if policy not in ('guarded', 'unguarded', 'never_merge'):
        raise ValueError(policy)
    limits = list(thresholds)
    if policy == 'never_merge':
        limits[2] = float('inf')
    state, audit = decode(observation, scores, limits, root_type, safe=policy != 'unguarded')
    audit['policy'] = policy
    if policy == 'never_merge':
        assert all(e['edit']['operation'] != 'MERGE_ALIASES' for e in audit['edits'])
        assert all(o == c for o, c in state.clusters.items())
    return state, audit
