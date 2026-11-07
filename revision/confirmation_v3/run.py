"""Freeze, prepare unused root histories, and evaluate unchanged fitted models."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import time
import joblib
from threadpoolctl import threadpool_limits
from revision import cohorts
from revision.improvement_v2.run import ROOT, write, subset, verify as verify_v2
from revision.improvement_v2.method import actions, corrupt, score
from revision.evaluation import damage_counts
from tgrepair.data import anonymize, load_graph, save_graph, sha256
from tgrepair.metrics import traceability, identity_pair_scores
from .method import apply_policy

HERE = Path(__file__).resolve().parent
OLD = ROOT / 'revision/improvement_v2'
DOMAINS = ['bpi2017','bpi2019']
BLOCKS = 8
ARMS = ['real8','real8_generated']


def verify():
    p=json.loads((HERE/'protocol.json').read_text())
    for group in ['implementation','inputs','models']:
        for name,digest in p[group].items():
            assert sha256(ROOT/name)==digest, name
    verify_v2()
    return p


def freeze():
    if (HERE/'protocol.json').exists(): verify(); return
    verify_v2()
    files=[HERE/'run.py',HERE/'method.py',HERE/'analyze.py',ROOT/'revision/cohorts.py',
           ROOT/'revision/improvement_v2/analyze.py',ROOT/'revision/improvement_v2/method.py']
    files += list((ROOT/'tgrepair').glob('*.py'))
    inputs=[ROOT/'revision/docs/COHORT_PLAN_V1.json',ROOT/'revision/docs/NATIVE_COHORT_AMENDMENT_V1.json',
            OLD/'protocol.json',OLD/'analysis.json']
    models=[]
    for d in DOMAINS:
        inputs += [ROOT/'revision/reports'/f'cohorts_{d}_v1.json',ROOT/'revision/reports'/f'native_cohort_{d}_v1.json']
        manifest=json.loads(inputs[-2].read_text()); inputs.append(Path(manifest['source_path']))
        for seed in range(3):
            for budget in [16,64,256]:
                for arm in ARMS:
                    path=OLD/'models'/d/f'seed{seed}_budget{budget}_full_{arm}.joblib'
                    models.extend([path,path.with_suffix('.json'),path.with_suffix('.calibration.npz')])
    # Source paths can be outside this project. Path joining preserves absolute paths.
    key=lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    write(HERE/'protocol.json',{
        'status':'frozen_before_fresh_cohort_selection_and_outcomes',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'domains':DOMAINS,'blocks_per_domain':BLOCKS,'roots_per_block':128,
        'selection':'Extend original hash-ranked partition plan after development512, annotation1024, confirmation1024, expanded native2048; next1024 roots. Original feasibility and confirmation exclusions and shared-child rules remain. Check old raw partition hashes before any repair.',
        'block_assignment':'SHA256(confirmation_v3_block:<raw root>) sorted then consecutive128; independently anonymize each block with seed361000+block.',
        'model_seeds':[0,1,2],'budgets':[16,64,256],'precision_targets':[.95,.99],
        'models':'36 existing full-information fits and calibration thresholds reused without retraining, tuning or selection',
        'fault':'mixed; fresh seed362000+100*block+model_seed; same damaged graph for every arm,budget,threshold,policy within block/seed',
        'primary':'guarded, both arms, all budgets, both targets, all blocks/seeds',
        'safety':'budget64,target.99,both arms: unguarded,guarded,never_merge; never_merge keeps all guard/addition/atomic-root rules and prohibits every identity merge',
        'jobs':768,'no_edit_baselines':48,
        'hypotheses':{
            'H1':'On loans at budget64,q=.99, generated training increases mean full-event recovery with no increase in mean new harm relative to Real8.',
            'H2':'For each domain/arm at budget64,q=.99, guarded merging retains true identity pairs and improves mean history F1 or recovery over never_merge without increasing mean new harm.',
            'decision':'Report exact mean-direction replication and uncertainty separately. A statistically supported H1 requires paired block-bootstrap95% lower recovery difference>0 and upper harm difference<=0. H2 additionally requires lower F1 or recovery difference>0 and upper harm difference<=0. These are conditional descriptive intervals, not familywise-adjusted significance tests.',
            'secondary':'Complete budget grid, seed/block variation, unguarded comparator, never-merge identity opportunity cost. No threshold chosen by test outcomes.'},
        'uncertainty':'Average the three fixed fitted-model/fault-seed runs within each disjoint block; paired bootstrap eight blocks10000 draws with seed363000; conditional on existing models and same source processes. Do not count events or three seeds as independent population samples.',
        'limits':['Untouched target histories for this project, source schemas previously examined; not unseen institutions or natural-error validation.', 'Context resources/vendors may overlap prior cohorts and blocks.', 'Prior annotation/calibration budget unchanged; no extra verified fitting labels.', 'No new generated training examples.'],
        'implementation':{key(p):sha256(p) for p in files},
        'inputs':{key(p):sha256(p) for p in inputs},'models':{key(p):sha256(p) for p in models}})


def prepare(domain):
    protocol=verify(); dest=HERE/'data'/domain
    if (dest/'cohort.json').exists():
        verify_cohort(domain); return
    plan=copy.deepcopy(json.loads((ROOT/'revision/docs/COHORT_PLAN_V1.json').read_text()))
    plan['partition_sizes']['native_comparison']=2048
    plan['partition_sizes']['fresh_confirmation']=BLOCKS*128
    prior=json.loads((ROOT/'revision/reports'/f'cohorts_{domain}_v1.json').read_text())
    native=json.loads((ROOT/'revision/reports'/f'native_cohort_{domain}_v1.json').read_text())
    graphs,bundles,audit,types,root_type=getattr(cohorts,domain)(Path(prior['source_path']),plan)
    for part in ['development','annotation','confirmation','native_comparison']:
        expected=(native if part=='native_comparison' else prior)['partitions'][part]
        assert audit['partitions'][part]['root_ids_sha256']==expected['root_ids_sha256']
    raw=graphs['fresh_confirmation']; raw_bundles=bundles['fresh_confirmation']
    # Raw adapters group records by trace; graph validation requires time order.
    raw.events.sort(key=lambda e:(e.time,e.uid))
    old_targets=set().union(*(set(o for o,t in g.objects.items() if t in types) for p,g in graphs.items() if p!='fresh_confirmation'))
    fresh_targets={o for o,t in raw.objects.items() if t in types}
    assert not old_targets & fresh_targets
    ranked=sorted(raw_bundles,key=lambda r:hashlib.sha256(('confirmation_v3_block:'+r).encode()).hexdigest())
    manifest={'domain':domain,'protocol_sha256':sha256(HERE/'protocol.json'),'audit':audit,
              'earlier_partition_hashes_verified':True,'old_target_overlap':0,
              'target_types':sorted(types),'root_type':root_type,'blocks':[]}
    for b in range(BLOCKS):
        roots=ranked[b*128:(b+1)*128]; assert len(roots)==128
        graph=subset(raw,raw_bundles,roots)
        graph,mapping,_=anonymize(graph,361000+b,return_maps=True)
        folder=dest/f'block{b}'; save_graph(graph,folder/'clean.json.gz')
        mapped={mapping[r]:sorted(mapping[o] for o in raw_bundles[r]) for r in roots}
        write(folder/'bundles.json',mapped)
        # Raw IDs are provenance only. No learner reads this file.
        write(folder/'source_roots.json',roots)
        manifest['blocks'].append({'block':b,'events':len(graph.events),'roots':128,
            'files':{n:sha256(folder/n) for n in ['clean.json.gz','bundles.json','source_roots.json']}})
    write(dest/'cohort.json',manifest)
    print(json.dumps({'prepared':domain,'events':[b['events'] for b in manifest['blocks']], 'overlap':0}),flush=True)


def verify_cohort(domain):
    manifest=json.loads((HERE/'data'/domain/'cohort.json').read_text())
    assert manifest['protocol_sha256']==sha256(HERE/'protocol.json')
    for b in manifest['blocks']:
        for name,digest in b['files'].items(): assert sha256(HERE/'data'/domain/f"block{b['block']}"/name)==digest
    return manifest


def evaluate(obs,reference,types,root_type,scores,meta,precision,policy,dest,pins):
    if (dest/'result.json').exists():
        result=json.loads((dest/'result.json').read_text()); assert result['pins']==pins
        for name,digest in result['artifacts'].items(): assert sha256(dest/name)==digest
        return result
    start=time.perf_counter(); cpu=time.process_time()
    state,audit=apply_policy(obs,scores,meta['thresholds'][str(precision)]['values'],root_type,policy)
    result={'pins':pins,'history':traceability(state.graph,reference,types,state.clusters),
            'identity':identity_pair_scores(state.graph,reference,types,state.clusters),
            'events':damage_counts(obs.graph,state.graph,reference,types,state.clusters),
            'seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu}
    save_graph(state.graph,dest/'repaired.json.gz'); write(dest/'clusters.json',state.clusters); write(dest/'audit.json',audit)
    while state.log: state.undo()
    assert state.graph==obs.graph and all(o==c for o,c in state.clusters.items())
    result['undo_verified']=True
    result['artifacts']={n:sha256(dest/n) for n in ['repaired.json.gz','clusters.json','audit.json']}
    write(dest/'result.json',result)
    return result


def run_domain(domain):
    verify(); manifest=verify_cohort(domain); start=time.perf_counter(); cpu=time.process_time()
    types=set(manifest['target_types']); root_type=manifest['root_type']; rows=[]
    output=HERE/'results'/domain; output.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=4):
        for b in range(BLOCKS):
            clean=load_graph(HERE/'data'/domain/f'block{b}'/'clean.json.gz')
            for seed in range(3):
                graph,reference=corrupt(clean,types,362000+100*b+seed,'mixed')
                folder=HERE/'data'/domain/f'block{b}'/f'seed{seed}'
                save_graph(graph,folder/'observed.json.gz'); write(folder/'reference.json',reference)
                t=time.perf_counter(); c=time.process_time(); obs=actions(graph,types)
                cost={'seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-c,'actions':len(obs.edits)}
                write(folder/'features.json',cost)
                noedit={'history':traceability(graph,reference,types),
                        'events':damage_counts(graph,graph,reference,types,{o:o for o in graph.objects})}
                write(folder/'no_edit.json',noedit)
                for budget in [16,64,256]:
                    for arm in ARMS:
                        path=OLD/'models'/domain/f'seed{seed}_budget{budget}_full_{arm}.joblib'
                        models=joblib.load(path); meta=json.loads(path.with_suffix('.json').read_text())
                        assert sha256(path)==meta['sha256']
                        t=time.perf_counter(); c=time.process_time(); scores=score(models,obs)
                        scoring={'seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-c}
                        conditions=[(q,'guarded') for q in [.95,.99]]
                        if budget==64: conditions += [(.99,p) for p in ['unguarded','never_merge']]
                        for q,policy in conditions:
                            job=f'block{b}_s{seed}_b{budget}_{arm}_q{q}_{policy}'
                            pins={'domain':domain,'block':b,'seed':seed,'budget':budget,'arm':arm,'precision':q,'policy':policy,
                                  'protocol_sha256':sha256(HERE/'protocol.json'),'cohort_sha256':sha256(HERE/'data'/domain/'cohort.json'),
                                  'model':str(path.relative_to(ROOT)),'model_sha256':sha256(path),
                                  'observed_sha256':sha256(folder/'observed.json.gz'),'reference_sha256':sha256(folder/'reference.json')}
                            result=evaluate(obs,reference,types,root_type,scores,meta,q,policy,output/job,pins)
                            rows.append({'job':job,'result_sha256':sha256(output/job/'result.json'),'features':cost,'scoring':scoring})
                        print(json.dumps({'done':domain,'block':b,'seed':seed,'budget':budget,'arm':arm,'jobs':len(rows)}),flush=True)
                write(output/'progress.json',{'evaluations':rows})
                del obs,scores
    verify()
    write(output/'summary.json',{'status':'completed','evaluations':rows,'seconds':time.perf_counter()-start,
        'cpu_seconds':time.process_time()-cpu,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'threads':4})


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['freeze','prepare','run']); p.add_argument('--domain',choices=DOMAINS)
    a=p.parse_args()
    if a.stage=='freeze': freeze()
    elif a.stage=='prepare': prepare(a.domain)
    else: run_domain(a.domain)


if __name__=='__main__': main()
