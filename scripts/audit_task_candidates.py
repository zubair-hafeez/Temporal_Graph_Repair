"""Evaluator-only candidate coverage; references never enter retrieval."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import load_graph,sha256
from tgrepair.entity_features import EntityIndex
from tgrepair.matching import ObservedIndex
from tgrepair.provenance import snapshot_sources


def coverage(graph,reference,types,event_budget=2048,seed=2027):
    index=ObservedIndex(graph);identity=reference['identity']
    selected=random.Random(seed).sample(graph.events,min(event_budget,len(graph.events)))
    by_type={t:{'missing_true_participations':0,'retrievable_missing_participations':0,
                'wrong_observed_participations':0,'retrievable_wrong_participations':0} for t in sorted(types)}
    for event in selected:
        for typ in sorted(types):
            candidates=index.candidates(event.uid,typ,128,allow_current=True)
            candidate_truth={identity[o] for o in candidates}
            truth={o for o in reference['event_participation'][event.uid] if graph.objects[o]==typ}
            current={o for o in event.participants if graph.objects[o]==typ}
            missing=truth-{identity[o] for o in current}
            wrong={o for o in current if identity[o] not in truth}
            row=by_type[typ]
            row['missing_true_participations']+=len(missing)
            row['retrievable_missing_participations']+=len(missing&candidate_truth)
            row['wrong_observed_participations']+=len(wrong)
            row['retrievable_wrong_participations']+=len(wrong&set(candidates))
    groups=defaultdict(list)
    for obj,true in identity.items():
        if graph.objects[obj] in types:groups[true].append(obj)
    # Positive labels are constructed after observation-only retrieval.
    entities=EntityIndex(graph,types);pairs=set(entities.pairs(64))
    truth_pairs={tuple(sorted(pair)) for ids in groups.values() for pair in combinations(ids,2)}
    both_observed={pair for pair in truth_pairs if all(o in entities.profile for o in pair)}
    for row in by_type.values():
        row['missing_participation_recall']=row['retrievable_missing_participations']/row['missing_true_participations'] if row['missing_true_participations'] else None
    return {'sampled_events':len(selected),'sample_seed':seed,'event_sampling':'uniform without reference/fault selection',
            'participant_budget':128,'pair_budget':64,'participation_by_type':by_type,
            'identity':{'candidate_pairs':len(pairs),'true_pairs':len(truth_pairs),'retrievable_true_pairs':len(pairs&truth_pairs),
                        'recall_all_true_pairs':len(pairs&truth_pairs)/len(truth_pairs) if truth_pairs else None,
                        'true_pairs_with_both_handles_observed':len(both_observed),
                        'recall_both_observed':len(pairs&both_observed)/len(both_observed) if both_observed else None},
            'target_catalog_handles_without_history':sum(graph.objects[o] in types and not index.degree[o] for o in graph.objects)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=ROOT/args.task;destination=ROOT/'reports'/f'{args.output}.json'
    if destination.exists():raise FileExistsError(destination)
    task=json.loads((folder/'task.json').read_text());graph=load_graph(folder/'observed.json.gz')
    reference=json.loads((folder/'reference.json').read_text());start=time.monotonic()
    if sha256(folder/'observed.json.gz')!=task['observation_sha256'] or sha256(folder/'reference.json')!=task['reference_sha256']:
        raise ValueError('Task artifact hash mismatch')
    report={'status':'evaluator_only_initial_candidate_audit','task_hash':sha256(folder/'task.json'),
            'source_snapshot':snapshot_sources(ROOT),'coverage':coverage(graph,reference,set(task['target_types'])),
            'seconds':time.monotonic()-start,'limits':['Initial observation only; later repair passes can change candidate coverage.',
                'Catalog availability and candidate recall do not imply that an identity is identifiable from observed evidence.']}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['coverage']),flush=True)


if __name__=='__main__':main()
