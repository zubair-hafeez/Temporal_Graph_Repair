"""Audit every prediction and calculate prespecified paired block uncertainty."""
from collections import defaultdict
import csv
import json
import numpy as np
from tgrepair.data import load_graph, sha256
from tgrepair.edits import Edit, RepairState
from revision.improvement_v2.analyze import independent_metrics
from revision.improvement_v2.run import write
from .run import HERE, ROOT, DOMAINS, BLOCKS, ARMS, verify, verify_cohort

FIELDS=['f1','exact','recovery','damage','recovery_rate','damage_rate','correct_pairs','false_pairs']
KEYS=['domain','budget','arm','precision','policy']


def contrast(records,domain,budget,q,left,right):
    deltas=[]
    for b in range(BLOCKS):
        differences=[]
        for seed in range(3):
            select=lambda arm,policy: next(r for r in records if (r['domain'],r['block'],r['seed'],r['budget'],r['precision'],r['arm'],r['policy'])==(domain,b,seed,budget,q,arm,policy))
            a=select(*left); z=select(*right)
            differences.append([a[k]-z[k] for k in FIELDS])
        deltas.append(np.mean(differences,axis=0))
    deltas=np.array(deltas)
    rng=np.random.default_rng(363000)
    boot=np.mean(deltas[rng.integers(0,BLOCKS,size=(10000,BLOCKS))],axis=1)
    return {'domain':domain,'budget':budget,'precision':q,'left':list(left),'right':list(right),
            'blocks':BLOCKS,'differences_by_block':deltas.tolist(),
            'metrics':{k:{'mean':float(deltas[:,i].mean()),'ci95':np.quantile(boot[:,i],[.025,.975]).tolist(),
                          'positive_blocks':int((deltas[:,i]>0).sum()),'negative_blocks':int((deltas[:,i]<0).sum())} for i,k in enumerate(FIELDS)}}


def main():
    verify(); records=[]; checked={}; baseline=[]
    for domain in DOMAINS:
        manifest=verify_cohort(domain); types=set(manifest['target_types'])
        summary_path=HERE/'results'/domain/'summary.json'; summary=json.loads(summary_path.read_text())
        assert summary['status']=='completed' and len(summary['evaluations'])==384
        expected={(b,s,budget,arm,q,'guarded') for b in range(BLOCKS) for s in range(3) for budget in [16,64,256] for arm in ARMS for q in [.95,.99]}
        expected|={(b,s,64,arm,.99,p) for b in range(BLOCKS) for s in range(3) for arm in ARMS for p in ['unguarded','never_merge']}
        actual=set(); cache={}
        checked[str(summary_path.relative_to(HERE))]=sha256(summary_path)
        checked[f'data/{domain}/cohort.json']=sha256(HERE/'data'/domain/'cohort.json')
        for row in summary['evaluations']:
            folder=summary_path.parent/row['job']; result_path=folder/'result.json'
            assert sha256(result_path)==row['result_sha256']
            checked[str(result_path.relative_to(HERE))]=sha256(result_path)
            result=json.loads(result_path.read_text()); p=result['pins']; b=p['block']; seed=p['seed']
            actual.add((b,seed,p['budget'],p['arm'],p['precision'],p['policy']))
            assert p['protocol_sha256']==sha256(HERE/'protocol.json')
            assert p['model_sha256']==sha256(ROOT/p['model'])
            if (b,seed) not in cache:
                task=HERE/'data'/domain/f'block{b}'/f'seed{seed}'
                graph=load_graph(task/'observed.json.gz'); ref=json.loads((task/'reference.json').read_text())
                for n in ['observed.json.gz','reference.json','no_edit.json']:
                    checked[str((task/n).relative_to(HERE))]=sha256(task/n)
                assert sha256(task/'observed.json.gz')==p['observed_sha256']
                assert sha256(task/'reference.json')==p['reference_sha256']
                cache[b,seed]=graph,ref
                baseline.append({'domain':domain,'block':b,'seed':seed,**json.loads((task/'no_edit.json').read_text())})
            graph,ref=cache[b,seed]
            for n,digest in result['artifacts'].items():
                assert sha256(folder/n)==digest
                checked[str((folder/n).relative_to(HERE))]=digest
            repaired=load_graph(folder/'repaired.json.gz'); clusters=json.loads((folder/'clusters.json').read_text())
            audit=json.loads((folder/'audit.json').read_text()); replay=RepairState(graph)
            replay.apply_batch([Edit(**item['edit']) for item in audit['edits']])
            assert replay.graph==repaired and replay.clusters==clusters
            while replay.log: replay.undo()
            assert replay.graph==graph
            m=independent_metrics(graph,repaired,ref,types,clusters)
            assert abs(m['f1']-result['history']['f1'])<1e-12
            assert abs(m['exact']-result['history']['exact_history_fraction'])<1e-12
            for k,v in [('damage','expanded_new_damage'),('recovery','expanded_repaired_events'),('correct_before','originally_correct_events'),('incorrect_before','originally_incorrect_events')]: assert m[k]==result['events'][v]
            assert m['correct_pairs']==result['identity']['correct_pairs']
            assert m['false_pairs']==result['identity']['predicted_pairs']-result['identity']['correct_pairs']
            if p['policy']=='never_merge': assert m['correct_pairs']==m['false_pairs']==0 and all(o==c for o,c in clusters.items())
            records.append({k:p[k] for k in ['domain','block','seed','budget','arm','precision','policy']} | m |
                           {'recovery_rate':m['recovery']/max(1,m['incorrect_before']), 'damage_rate':m['damage']/max(1,m['correct_before']),
                            'decode_metric_seconds':result['seconds'],'scoring_seconds':row['scoring']['seconds'],'feature_seconds':row['features']['seconds']})
        assert actual==expected
    groups=defaultdict(list)
    for r in records: groups[tuple(r[k] for k in KEYS)].append(r)
    aggregates=[]
    for key,rows in groups.items():
        assert len(rows)==24
        aggregates.append(dict(zip(KEYS,key))|{'n':24,**{k:{'mean':float(np.mean([r[k] for r in rows])), 'min':float(min(r[k] for r in rows)), 'max':float(max(r[k] for r in rows))} for k in FIELDS}})
    paired=[]
    for d in DOMAINS:
        for budget in [16,64,256]:
            for q in [.95,.99]: paired.append(contrast(records,d,budget,q,('real8_generated','guarded'),('real8','guarded')))
        for arm in ARMS:
            for other in ['never_merge','unguarded']: paired.append(contrast(records,d,64,.99,(arm,'guarded'),(arm,other)))
    with (HERE/'per_trial.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    write(HERE/'analysis.json',{'status':'all_768_conditions_independently_audited','conditions':len(records),
        'aggregates':aggregates,'contrasts':paired,'no_edit_baselines':baseline,'checked_files':checked,
        'analysis_code_sha256':sha256(__file__),'protocol_sha256':sha256(HERE/'protocol.json')})
    print(json.dumps({'conditions':len(records),'contrasts':len(paired)}),flush=True)


if __name__=='__main__': main()
