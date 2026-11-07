"""Independently check saved block coverage against raw selection manifests."""
import hashlib
import json
from tgrepair.data import load_graph, sha256
from tgrepair.confirmation_sources import _list_hash
from .run import HERE, ROOT, DOMAINS, verify, verify_cohort, write


def main():
    verify(); results=[]
    for domain in DOMAINS:
        manifest=verify_cohort(domain); types=set(manifest['target_types']); roots=[]; target_hashes=[]
        for b in manifest['blocks']:
            folder=HERE/'data'/domain/f"block{b['block']}"
            raw=json.loads((folder/'source_roots.json').read_text()); assert len(raw)==len(set(raw))==128
            roots.extend(raw)
            graph=load_graph(folder/'clean.json.gz'); bundles=json.loads((folder/'bundles.json').read_text())
            targets={o for o,t in graph.objects.items() if t in types}
            assert len(bundles)==128
            members=[o for values in bundles.values() for o in values]
            assert len(members)==len(set(members)) and set(members)==targets
            assert all(sum(graph.objects[o]==manifest['root_type'] for o in e.participants)==1 for e in graph.events)
            assert len(graph.events)==b['events']
            target_hashes.append(sha256(folder/'bundles.json'))
        assert len(roots)==len(set(roots))==1024
        ranked=sorted(roots,key=lambda r:hashlib.sha256(('confirmation_v3_block:'+r).encode()).hexdigest())
        assert roots==ranked
        raw_names=[r.split(':',1)[1] for r in roots] if domain=='bpi2017' else roots
        assert _list_hash(raw_names)==manifest['audit']['partitions']['fresh_confirmation']['root_ids_sha256']
        prior=json.loads((ROOT/'revision/reports'/f'cohorts_{domain}_v1.json').read_text())
        native=json.loads((ROOT/'revision/reports'/f'native_cohort_{domain}_v1.json').read_text())
        for part in ['development','annotation','confirmation','native_comparison']:
            expected=(native if part=='native_comparison' else prior)['partitions'][part]
            assert manifest['audit']['partitions'][part]['root_ids_sha256']==expected['root_ids_sha256']
        assert manifest['audit']['root_and_child_disjointness_verified'] and manifest['old_target_overlap']==0
        assert manifest['audit']['original_confirmation_root_ids_sha256']==prior['original_confirmation_root_ids_sha256']
        assert manifest['audit']['excluded_root_ids_sha256']==prior['excluded_root_ids_sha256']
        results.append({'domain':domain,'raw_roots':1024,'unique_events':sum(b['events'] for b in manifest['blocks']),
                        'cohort_sha256':sha256(HERE/'data'/domain/'cohort.json'),
                        'checks':['raw block union equals frozen selected cohort','block ordering matches declared hash',
                                  'all prior partition hashes unchanged','original excluded roots unchanged',
                                  'target bundles do not overlap within blocks','exactly one native root per retained event']})
    write(HERE/'freshness.json',{'status':'passed','domains':results,'script_sha256':sha256(__file__),
        'scope':'Disjoint target histories in this project; known source schemas and potentially shared context.'})
    print(json.dumps(results))


if __name__=='__main__': main()
