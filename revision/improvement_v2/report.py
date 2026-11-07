"""Readable complete follow-up results, including negative findings and costs."""
import json

from .run import HERE


def main():
    analysis = json.loads((HERE / 'analysis.json').read_text())
    assert analysis['conditions'] == 156
    rows = analysis['aggregates']
    def get(domain, budget=64, arm='real8_generated', mode='full', precision=.95, variant='mixed', safe=True):
        return next(r for r in rows if r['domain'] == domain and r['budget'] == budget and r['arm'] == arm
                    and r['mode'] == mode and r['precision'] == precision and r['variant'] == variant and r['safe'] == safe)
    lines = ['REPEATED-CORRUPTION FOLLOW-UP: COMPLETE RESULTS', '',
             '60 fitted models; 156 fixed-grid repair conditions; 2 sources; 3 seeds.',
             'All edits, undo records, calibration scores/thresholds and main metrics audited.',
             'These are reused source cohorts, not independent confirmation or natural errors.',
             'Values below are three-seed means. F1 is a percentage; R/H are recovered/damaged events.', '',
             'MATCHED REAL-HISTORY BUDGETS',
             'Source  Budget  q     Real8: F1 / R / H       +Generated: F1 / R / H']
    for domain in ['bpi2017', 'bpi2019']:
        for budget in [16, 64, 256]:
            for p in [.95, .99]:
                a, b = get(domain, budget, 'real8', precision=p), get(domain, budget, precision=p)
                lines.append(f"{domain} {budget:4} {p:.2f}   " + '     '.join(
                    f"{r['f1']['mean']*100:5.2f} / {r['recovery']['mean']:6.1f} / {r['damage']['mean']:5.1f}" for r in [a, b]))
    lines += ['', 'SEED-LEVEL RECOVERY/DAMAGE DOMINANCE AT IDENTICAL CALIBRATION TARGETS',
              'Dominance requires at least as much recovery and no more damage, with one strict inequality.',
              'Identical calibration precision targets do not guarantee identical test harm.']
    for pair in analysis['paired']:
        lines.append(f"{pair['domain']} B={pair['budget']} q={pair['precision']}: "
                     f"generated dominates {pair['generated_dominates_seeds']}/3; real dominates {pair['real_dominates_seeds']}/3")
    lines += ['', 'INFORMATION ABLATIONS: GENERATED ARM, BUDGET64, q=.95']
    for domain in ['bpi2017', 'bpi2019']:
        for mode in ['full', 'no_time', 'no_context', 'no_attributes', 'no_history']:
            r = get(domain, mode=mode)
            lines.append(f"{domain} {mode:14} F1={r['f1']['mean']*100:.2f}, R={r['recovery']['mean']:.1f}, H={r['damage']['mean']:.1f}")
    lines += ['', 'GUARD COMPARISON: BUDGET64, q=.99, SAME FITTED MODELS AND THRESHOLDS',
              'Both retain atomic root substitutions; only tie/merge guards change.']
    for domain in ['bpi2017', 'bpi2019']:
        for arm in ['real8', 'real8_generated']:
            for safe in [False, True]:
                r = get(domain, arm=arm, precision=.99, safe=safe)
                lines.append(f"{domain} {arm:16} guarded={safe} R={r['recovery']['mean']:.1f}, H={r['damage']['mean']:.1f}, "
                             f"true_pairs={r['correct_pairs']['mean']:.1f}, false_pairs={r['false_pairs']['mean']:.1f}")
    lines += ['', 'CLEAN AND KNOWN-FAMILY PIPELINE STRESS: BUDGET64, q=.95']
    for domain in ['bpi2017', 'bpi2019']:
        for variant in ['clean', 'pipeline']:
            for arm in ['real8', 'real8_generated']:
                r = get(domain, arm=arm, variant=variant)
                lines.append(f"{domain} {variant:8} {arm:16} F1={r['f1']['mean']*100:.2f}, R={r['recovery']['mean']:.1f}, H={r['damage']['mean']:.1f}")
    lines += ['', 'EMPTY-HISTORY AMBIGUITY CHALLENGE']
    ambiguity = [json.loads(path.read_text()) for path in (HERE / 'results').glob('*/ambiguity_*.json')]
    for safe in [False, True]:
        selected = [r for r in ambiguity if r['audit']['safe'] == safe]
        lines.append(f"guarded={safe}: {sum(r['unresolved'] for r in selected)}/{sum(r['events'] for r in selected)} events left unresolved; "
                     f"{sum(r['unsupported_edits'] for r in selected)} unsupported edits")
    lines += ['', 'MEASURED COMPUTE', 'Times reflect this host and concurrent workers, not a hardware-independent benchmark.']
    generation = json.loads((HERE / 'generation_benchmark.json').read_text())
    lines.append(f"CPU: {generation['cpu_model']}. Four fitting threads per worker; no GPU training.")
    lines.append(f"32-world execution+fault creation: {generation['generation_wall_seconds']:.3f}s wall / {generation['generation_cpu_seconds']:.3f}s CPU, excluding disk I/O.")
    lines.append(f"Shared full-information generated feature cache: {generation['feature_cache_times']['full']['seconds']:.2f}s wall.")
    for domain in ['bpi2017', 'bpi2019']:
        total = json.loads((HERE / 'results' / domain / 'summary.json').read_text())
        lines.append(f"{domain} full worker: {total['seconds']:.1f}s wall / {total['cpu_seconds']:.1f}s CPU; peak RSS {total['max_rss_kib']/1024**2:.2f} GiB.")
        for budget in [16, 64, 256]:
            for arm in ['real8', 'real8_generated']:
                r = get(domain, budget, arm)
                lines.append(f"  B={budget:3} {arm:16} mean fit {r['train_seconds']['mean']:.2f}s wall / {r['train_cpu_seconds']['mean']:.2f}s CPU; "
                             f"inference+metrics+feature construction {r['inference_seconds']['mean']:.2f}s wall")
    lines += ['', 'BOUNDARIES',
              'Budgets count complete reference histories, including held-out calibration histories.',
              'The added generated examples change training row count and weight distribution; compute is not matched.',
              'No timestamps removes time before retrieval; no-context retains target root/child incidence.',
              'No-history removes all incidence from evidence, retaining current links only to define candidate operations.',
              'The ambiguity challenge has indistinguishable empty histories; it does not cover every ambiguity mechanism.',
              'Pipeline faults are among training variants, so their stress test is not an unseen-family test.',
              'A fixed equal mixture and one bounded generator do not establish that all synthetic data is unnecessary.',
              'The user authorized skipping investigator-verified natural errors and actual human-time collection.',
              'No measured annotation-time saving, universal safety, causal generator-component effect or architecture superiority is claimed.',
              'Earlier results remain preserved and use a different cohort size/context/decoder; their scores are not directly comparable.',
              '', 'Reproduce with REPRODUCE.txt; exact per-trial values and ranges are in per_trial.csv and analysis.json.']
    (HERE / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
