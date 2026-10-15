"""Evaluator-only paired feature diagnosis; no output trains a repair model."""
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,sha256
from tgrepair.matching import ObservedIndex,FEATURE_NAMES
from tgrepair.action_features import ACTION_DIM
from tgrepair.provenance import snapshot_sources
from evaluate_edit_policy import thresholds_from_validation


def main():
    destination=ROOT/'reports/external_attribute_diagnostic_v1.json'
    if destination.exists():raise FileExistsError(destination)
    models=joblib.load(ROOT/'checkpoints/edit_feature_v2.joblib')
    thresholds,_=thresholds_from_validation('edit_feature_v2',.95)
    report={'status':'evaluator_only_development_diagnostic','source_snapshot':snapshot_sources(ROOT),
            'model_hash':sha256(ROOT/'checkpoints/edit_feature_v2.joblib'),
            'thresholds':thresholds,'limits':['256 randomly selected event queries per domain.',
                'Fixed native candidate pools isolate score changes when fields are hidden; this is not a separate calibrated model.'],
            'domains':{}}
    with threadpool_limits(limits=4):
        for domain in ['batteryvault','claimstream']:
            base=ROOT/'data/external_development_v1'/domain/'mixed'
            graph=load_graph(base/'observed.json.gz')
            structural=load_graph(ROOT/'data/external_structural_v1'/domain/'mixed'/'observed.json.gz')
            reference=json.loads((base/'reference.json').read_text())
            types=set(json.loads((base/'task.json').read_text())['target_types'])
            native_index=ObservedIndex(graph);structural_index=ObservedIndex(structural)
            rows=defaultdict(list);labels=[];ops=[];coverage=[]
            for event in random.Random(101).sample(graph.events,min(256,len(graph.events))):
                truth=set(reference['event_participation'][event.uid])
                present={reference['identity'][o] for o in event.participants}
                for typ in sorted(types):
                    candidates=native_index.candidates(event.uid,typ,128,allow_current=True)
                    expected={o for o in truth if graph.objects[o]==typ}
                    if expected:coverage.append(len(expected&{reference['identity'][o] for o in candidates})/len(expected))
                    for view,index in [('native',native_index),('structural',structural_index)]:
                        features=index.features(event.uid,candidates,exclude_query=True)
                        values=np.zeros((len(candidates),ACTION_DIM),dtype=np.float32)
                        values[:,:len(FEATURE_NAMES)]=features
                        values[:,25]=index.type_kind_count[typ][event.kind]/max(1,index.kind_count[event.kind])
                        values[:,26]=sum(graph.objects[o]==typ for o in event.participants)
                        rows[view].append(np.clip(values,-10,10))
                    for obj in candidates:
                        op=int(obj in event.participants)
                        label=(reference['identity'][obj] not in truth) if op else (reference['identity'][obj] in truth-present)
                        labels.append(label);ops.append(op)
            y=np.asarray(labels);operations=np.asarray(ops);result={}
            for view,parts in rows.items():
                x=np.concatenate(parts);detail={}
                for op in [0,1]:
                    mask=operations==op;p=models[op].predict_proba(x[mask])[:,1];target=y[mask]
                    accepted=p>=thresholds[op]
                    detail[str(op)]={'actions':int(mask.sum()),'positives':int(target.sum()),
                                     'average_precision':float(average_precision_score(target,p)),
                                     'accepted_correct':int((accepted&target).sum()),'accepted_wrong':int((accepted&~target).sum()),
                                     'positive_score_quantiles':np.quantile(p[target],[.1,.5,.9,.99]).tolist() if target.any() else [],
                                     'negative_score_quantiles':np.quantile(p[~target],[.1,.5,.9,.99]).tolist(),
                                     'positive_mean_features':dict(zip(FEATURE_NAMES,x[mask][target,:len(FEATURE_NAMES)].mean(0).astype(float).tolist())) if target.any() else {}}
                result[view]=detail
            stability=defaultdict(list)
            for obj,fields in native_index.attribute_values.items():
                if graph.objects[obj] in types:
                    for field,counts in fields.items():
                        if sum(counts.values())>=3:stability[field].append(max(counts.values())/sum(counts.values()))
            result['observed_field_modal_share']={k:{'mean':float(np.mean(v)),'entities':len(v)} for k,v in stability.items()}
            result['mean_native_candidate_recall']=float(np.mean(coverage))
            report['domains'][domain]=result
            destination.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'domain':domain,'recall':result['mean_native_candidate_recall'],
                              'native':{k:{q:v for q,v in a.items() if q in ['average_precision','accepted_correct','accepted_wrong']} for k,a in result['native'].items()},
                              'structural':{k:{q:v for q,v in a.items() if q in ['average_precision','accepted_correct','accepted_wrong']} for k,a in result['structural'].items()}}),flush=True)


if __name__=='__main__':main()
