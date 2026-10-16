"""Check that attribute scenarios preserve execution ancestry and fault labels."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256


def main():
    destination=ROOT/'reports/attribute_scenarios_audit_v1.json'
    if destination.exists():raise FileExistsError(destination)
    report={'status':'paired_attribute_scenario_integrity_check','collections':{},
            'count_note':'The v5 generation manifest field world_events counts visible clean learner events, unlike the v4 full execution count. Audit reports use visible_events explicitly.'}
    for split in ['train','validation']:
        parent=ROOT/'data/generated'/f'workflow_{split}_v4'
        derived=ROOT/'data/generated'/f'workflow_{split}_attributes_v5'
        source_manifest=json.loads((parent/'dataset_manifest.json').read_text())
        target_manifest=json.loads((derived/'dataset_manifest.json').read_text())
        assert target_manifest['parent_manifest_hash']==sha256(parent/'dataset_manifest.json')
        assert len(source_manifest['episodes'])==len(target_manifest['episodes'])
        rows=0;fields_changed=0;visible=0
        for source,view in zip(source_manifest['episodes'],target_manifest['episodes']):
            assert (source['id'],source['group'],source['execution_hash'],source['recipe'])==(
                    view['id'],view['group'],view['execution_hash'],view['recipe'])
            by_origin={}
            for variant in ['clean','mixed']:
                a=parent/source['id']/variant;b=derived/source['id']/variant
                original=json.loads((a/'learner.json').read_text());transformed=json.loads((b/'learner.json').read_text())
                reference=json.loads((a/'reference.json').read_text())
                assert reference==json.loads((b/'reference.json').read_text())
                assert original['objects']==transformed['objects']
                assert len(original['events'])==len(transformed['events'])
                if variant=='clean':visible+=len(original['events'])
                for e,f in zip(original['events'],transformed['events']):
                    assert {k:v for k,v in e.items() if k!='attributes'}=={k:v for k,v in f.items() if k!='attributes'}
                    origin=reference['event_origin'][f['uid']]
                    if variant=='clean':by_origin[origin]=f['attributes']
                    else:assert by_origin[origin]==f['attributes']
                    fields_changed+=int(e['attributes']!=f['attributes']);rows+=1
        report['collections'][split]={'episodes':len(target_manifest['episodes']),'checked_variant_events':rows,
                                     'visible_events':visible,'changed_attribute_records':fields_changed,
                                     'parent_hash':sha256(parent/'dataset_manifest.json'),
                                     'derived_hash':sha256(derived/'dataset_manifest.json'),
                                     'facts_and_fault_references_identical':True,'paired_attributes_identical':True}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['collections']))


if __name__=='__main__':main()
