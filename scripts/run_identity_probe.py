"""Development identity and application gate; no final targets are read."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import load_graph, sha256
from tgrepair.corruption import mixed_faults
from tgrepair.identity_matching import alias_candidates, match_aliases, attribute_variant_clusters
from tgrepair.metrics import identity_pair_scores, traceability


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['bpi2012_fit','bpi2012_tune','logistics','bpi2012_tune_attributes'], default='bpi2012_tune')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='identity_probe_v1')
    args = p.parse_args()
    dest = ROOT / 'reports' / (args.output + '.json')
    if dest.exists():
        raise FileExistsError(dest)
    path = ROOT / 'data/development' / (args.dataset + '.json.gz')
    graph = load_graph(path)
    target_types = set(graph.objects.values()) - {'Resource','Forklift','Truck','Vehicle'}
    start = time.monotonic()
    report = {'status':'development_identity_diagnostic','dataset':args.dataset,'seed':args.seed,
              'source_hash':sha256(path), 'runs':[],
              'limits':['No observation attributes or object-object links in this track.',
                        'Nonoverlap is a baseline assumption; it is not enforced on real business behavior.',
                        'Mixed wrong participants are random same-type alternatives; hard stale-map faults remain pending.']}
    for name, missing, wrong in [('fragmentation_only',0,0),('mixed',0.1,0.1)]:
        observed, truth = mixed_faults(graph,args.seed,target_types,missing_rate=missing,wrong_rate=wrong,fragment_rate=0.25)
        no_change = {o:o for o in observed.objects}
        use_attributes = any(e.attributes for e in observed.events)
        edges = alias_candidates(observed,target_types,use_attributes=use_attributes)
        true_pairs = {frozenset(pair) for pair in truth['fragmentations']}
        reachable = {frozenset((a,b)) for a,b,_ in edges}
        run = {'condition':name,'config':truth['config'],'fragmentations':len(true_pairs),
               'alias_candidate_recall':len(true_pairs & reachable)/len(true_pairs) if true_pairs else None,
               'missing_links':len(truth['missing']),'wrong_links':len(truth['wrong']),
               'no_change':{'traceability':traceability(observed,truth,target_types),
                            'identity':identity_pair_scores(observed,truth,target_types,no_change)},
               'comparisons':[]}
        if use_attributes:
            clusters = attribute_variant_clusters(observed, target_types)
            run['attribute_variant_adaptation'] = {
                'traceability':traceability(observed,truth,target_types,clusters),
                'identity':identity_pair_scores(observed,truth,target_types,clusters)}
        for threshold in [0.25,0.5,0.75,0.9]:
            for global_assignment in [False,True]:
                clusters = match_aliases(observed,edges,target_types,threshold,global_assignment)
                run['comparisons'].append({'threshold':threshold,'global_assignment':global_assignment,
                                          'traceability':traceability(observed,truth,target_types,clusters),
                                          'identity':identity_pair_scores(observed,truth,target_types,clusters)})
        report['runs'].append(run)
        report['seconds'] = time.monotonic()-start
        dest.write_text(json.dumps(report,indent=2)+'\n')
        best = max(run['comparisons'],key=lambda x:x['traceability']['f1'])
        print(json.dumps({'condition':name,'fragmentations':run['fragmentations'],
                          'alias_candidate_recall':run['alias_candidate_recall'],
                          'no_change_f1':run['no_change']['traceability']['f1'],
                          'best_development_setting':{k:best[k] for k in ['threshold','global_assignment']},
                          'best_f1':best['traceability']['f1'],'identity':best['identity']}),flush=True)


if __name__=='__main__':
    main()
