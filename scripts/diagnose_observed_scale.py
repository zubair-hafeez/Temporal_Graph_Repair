"""Observed-only population/context diagnostic; no reference labels are read."""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,sha256
from tgrepair.factory_plugin import learner_graph


def profile(graph,types):
    histories=defaultdict(list);contexts=defaultdict(set);peers=defaultdict(set)
    for event in graph.events:
        targets=[o for o in event.participants if graph.objects[o] in types]
        other=[o for o in event.participants if graph.objects[o] not in types]
        for obj in targets:
            histories[obj].append(event)
            contexts[obj].update(other)
        for context in other:peers[context].update(targets)
    def q(values):return np.quantile(list(values),[.1,.5,.9,.99]).astype(float).tolist()
    return {'catalog_target_handles':sum(t in types for t in graph.objects.values()),
            'observed_target_handles':len(histories),'context_handles':len(peers),
            'events_per_target':q(len(v) for v in histories.values()),
            'contexts_per_target':q(len(contexts[o]) for o in histories),
            'targets_per_context':q(len(v) for v in peers.values())}


def main():
    destination=ROOT/'reports/observed_scale_diagnostic_v1.json'
    if destination.exists():raise FileExistsError(destination)
    path=ROOT/'data/edit_learning/selfsupervised_controls_v1/imperfect_input.json.gz'
    source=profile(load_graph(path),{'Application'})
    folder=ROOT/'data/generated/workflow_train_v4'
    manifest=json.loads((folder/'dataset_manifest.json').read_text())
    episodes=random.Random(91).sample([e for e in manifest['episodes'] if e['projection']=='flat'],32)
    generated=[]
    for episode in episodes:
        learner=json.loads((folder/episode['id']/'mixed/learner.json').read_text())
        generated.append({'episode':episode['id'],**profile(learner_graph(learner),set(learner['schema']['target_types']))})
    report={'status':'fit_only_observed_scale_diagnostic','quantiles':[.1,.5,.9,.99],
            'source_hash':sha256(path),'generated_manifest_hash':sha256(folder/'dataset_manifest.json'),
            'source_fit':source,'generated':generated,
            'generated_median_profile':{key:np.median([r[key] for r in generated],axis=0).tolist() for key in source},
            'contract':'Observed memberships and context only; no clean identities or final targets inspected.'}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'source_fit':source,'generated_median_profile':report['generated_median_profile']}))


if __name__=='__main__':main()
