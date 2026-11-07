"""Descriptive fitted-seed sensitivity of H1; does not change its frozen analysis."""
import csv
import json
import numpy as np
from .run import HERE, write
from tgrepair.data import sha256


def main():
    rows=list(csv.DictReader((HERE/'per_trial.csv').open())); seeds=[]
    for seed in range(3):
        record={'seed':seed}
        for arm in ['real8','real8_generated']:
            chosen=[r for r in rows if r['domain']=='bpi2017' and r['budget']=='64' and r['precision']=='0.99'
                    and r['policy']=='guarded' and r['seed']==str(seed) and r['arm']==arm]
            assert len(chosen)==8 and {int(r['block']) for r in chosen}==set(range(8))
            record[arm]={k:float(np.mean([float(r[k]) for r in chosen])) for k in ['recovery','damage','f1']}
        record['difference']={k:record['real8_generated'][k]-record['real8'][k] for k in ['recovery','damage','f1']}
        seeds.append(record)
    write(HERE/'seed_sensitivity.json',{'scope':'Descriptive decomposition of H1 across the three existing fitted-model/fault seeds. No new fit or hypothesis selection.',
        'per_trial_sha256':sha256(HERE/'per_trial.csv'),'script_sha256':sha256(__file__),'seeds':seeds,
        'positive_recovery_seed_means':sum(r['difference']['recovery']>0 for r in seeds),
        'negative_recovery_seed_means':sum(r['difference']['recovery']<0 for r in seeds)})
    print(json.dumps(seeds))


if __name__=='__main__': main()
