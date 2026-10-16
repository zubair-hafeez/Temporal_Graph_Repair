"""Record the concrete comparison plan; freezing is an explicit local step."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256


def make_protocol():
    prior=Path('/mnt/zubair/Temporal_Graph_Papers_2024-2026/analysis/new_idea_20261015')
    b17=json.loads((prior/'data_feasibility.json').read_text())['bpi2017']
    b19=json.loads((prior/'bpi2019_feasibility.json').read_text())
    source12=ROOT/'data/development/bpi2012_check_attributes.json.gz'
    sources={
        'bpi2012_check':{'path':str(source12),'sha256':sha256(source12),'target_types':['Application'],
                        'views':['denormalized','structural'],'role':'same-process confirmation'},
        'bpi2017':{'path':b17['source'],'sha256':b17['sha256'],'target_types':['Application','Offer'],
                   'views':['native','structural'],'role':'new process/system at the BPI2012 institution',
                   'adapter_options':{'cohort_size':1024,'audited_traces':500,'seed':91000}},
        'bpi2019':{'path':str(prior/'evidence/BPIC19_ocel2.zip'),'sha256':b19['sha256'],
                   'target_types':['PO','POItem'],'views':['native','structural'],
                   'role':'different real-source application family',
                   'adapter_options':{'cohort_size':1024,'audited_events':2000,'seed':91000}},
    }
    external=ROOT/'data/external/relational_ocel'
    hashes={line.split()[1]:line.split()[0] for line in (external/'SHA256SUMS').read_text().splitlines()}
    for name,types in {'forgeflow':['SalesOrder','Shipment'],'permitflow':['PermitApplication','Inspection'],
                       'procurechange':['PurchaseRequisition','Invoice'],'trialversion':['Participant','Sample']}.items():
        sources[name]={'path':str(external/(name+'.jsonocel.gz')),'sha256':hashes[name+'.jsonocel.gz'],
                       'target_types':types,'views':['native','structural'],
                       'role':'reserved scenario from one independent simulator collection','adapter_options':{'seed':91000}}
    models={}
    for name,kind,calibrations in [
        ('edit_feature_v2','feature',['edit_feature_v2_attributes_calibration_v3','edit_feature_v2_stable_calibration_v3']),
        ('edit_feature_attributes_v3','feature',['edit_feature_attributes_v3']),
        *[(f'edit_feature_{variant}_repeat{r}','feature',[f'edit_feature_{variant}_repeat{r}']) for r in [1,2] for variant in ['plain','attributes']],
        ('source_edit_learning_v2_source_only_256','feature',['source_edit_learning_v2_source_only_256','source256_generated_calibration_v3']),
        *[(f'edit_feature_no_{name}_v3','feature',[f'edit_feature_no_{name}_v3']) for name in ['temporal','context']],
        *[(f'edit_{name}_attributes_v3','neural',[f'edit_{name}_attributes_v3']) for name in ['sage','mlp','temporal']],
    ]:
        path=ROOT/'checkpoints'/(name+('.pt' if kind=='neural' else '.joblib'))
        models[name]={'kind':kind,'sha256':sha256(path),'calibrations':{
            cal:sha256(ROOT/'data/edit_learning'/f'{cal}_validation_scores.npz') for cal in calibrations}}
    adapters=['scripts/prepare_confirmation_tasks.py','tgrepair/confirmation_sources.py','tgrepair/json_stream.py',
              'tgrepair/data.py','tgrepair/external_ocel.py','tgrepair/corruption.py','tgrepair/pipeline_faults.py']
    implementation=['scripts/evaluate_external_policy.py','scripts/evaluate_edit_policy.py','tgrepair/action_features.py','tgrepair/matching.py',
                    'tgrepair/entity_features.py','tgrepair/decoding.py','tgrepair/edits.py','tgrepair/neural_policy.py',
                    'tgrepair/view_decoding.py','tgrepair/attribute_views.py','tgrepair/metrics.py','tgrepair/application_queries.py',
                    'tgrepair/confirmation_contract.py']
    adaptation_scripts=['scripts/prepare_target_selfsupervision.py','scripts/prepare_selfsupervised_control.py',
                        'scripts/prepare_source_edit_controls.py','scripts/train_selfsupervised_controls.py',
                        'scripts/train_edit_feature_policy.py','scripts/score_edit_feature_validation.py']
    return {'status':'draft_not_frozen','created_utc':datetime.now(timezone.utc).isoformat(),
            'task_root':'data/confirmation_v1','sources':sources,'models':models,
            'adapter_hashes':{p:sha256(ROOT/p) for p in adapters},
            'implementation_hashes':{p:sha256(ROOT/p) for p in implementation},
            'fault_families':['mixed','pipeline'],'fault_seeds':[91001,91002,91003],
            'generated_confirmation':{
                'original_collection':'workflow_confirmation_v1','attribute_collection':'workflow_confirmation_attributes_v1',
                'count':96,'world_seed':470000,'attribute_seed':1321000,'split':'test','diversity':'expanded',
                'decoder_contract':[8,16,32,.95],
                'script_hashes':{p:sha256(ROOT/p) for p in ['scripts/generate_workflow_pilot.py','scripts/generate_attribute_scenarios.py']},
                'comparison':'All three paired original/varied learners on the same 96 held-out mechanism-composition worlds.',
                'limit':'Compositional generalization within the same simulator, not independent application evidence.'},
            'decoder_contract':[4,128,64,.95],
            'primary_comparison':{
                'description':'Original versus varied-attribute generated training, paired workflows/calibration/fault seeds.',
                'pairs':[
                    {'replicate':0,'fault_seed':91001,'original':'edit_feature_v2','varied':'edit_feature_attributes_v3'},
                    {'replicate':1,'fault_seed':91002,'original':'edit_feature_plain_repeat1','varied':'edit_feature_attributes_repeat1'},
                    {'replicate':2,'fault_seed':91003,'original':'edit_feature_plain_repeat2','varied':'edit_feature_attributes_repeat2'}],
                'variant':'mixed','views':'native, or explicitly denormalized BPI2012 check',
                'replication_limit':'Three joint world-generation/fault trials; training and test-fault variance are not separately estimated.'},
            'secondary_comparisons':{
                'clean_preservation':'All three principal learners on each native clean counterpart.',
                'pipeline_stress':'All three principal learners on paired pipeline-fault seeds; unseen mechanism family in training.',
                'structural_view':'Original and varied learners, replicate 0 / seed 91001, attributes hidden for both.',
                'scoring_ablation':'Refitted no-temporal and no-context scoring controls on native mixed seed 91001; retrieval retained.',
                'decoding_ablation':'One pass and fixed sequencing using varied replicate 0 on native mixed seed 91001.',
                'neural_controls':'Matched GraphSAGE, MLP and temporal-message models on native mixed seed 91001.',
                'source_supervision':'BPI2012 source-only 256-history model with source-development calibration, all three mixed seeds; training model fixed and additional calibration labels disclosed.',
                'automatic_view':'Original model with stable-field view and its own generated calibration, native mixed seed 91001.'},
            'target_local_adaptation':{
                'description':'Self-supervised extra-corruption learning on each imperfect native mixed target, all three seeds, no original reference file read.',
                'script_hashes':{p:sha256(ROOT/p) for p in adaptation_scripts},
                'validation_by_fault_seed':{str(seed):{'name':name,'manifest_hash':sha256(ROOT/'data/edit_learning'/name/'manifest.json')}
                    for seed,name in zip([91001,91002,91003],['edit_validation_attributes_v3','edit_validation_attributes_repeat1','edit_validation_attributes_repeat2'])},
                'training_event_budget':8192,'validation_event_budget':2048,'time_partition':.7,
                'regime':'self_supervised','calibration':'same generated validation as the paired principal trial',
                'limits':'Transductive target access without clean target labels; time partitions can share entity handles.'},
            'primary_outcomes':['entity-history precision/recall/F1','exact-history fraction'],
            'secondary_outcomes':['identity-pair precision/recall and false pairs','initially correct-event preservation',
                                  'cancellation/exception evidence retrieval','first/last timestamp recovery','candidate coverage','runtime'],
            'reporting':['Retain every predeclared domain and failure; no post-outcome target selection.',
                         'Report per-process/type scores and three trial values/means/ranges, not event-level pseudo-replicate significance.',
                         'No unseen-organization claim for BPI2017; no real-data claim for the external simulations.',
                         'Controlled faults on source references do not prove naturally occurring error recovery.',
                         'No paper acceptance or universal recovery guarantee.']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args();protocol=make_protocol()
    destination=ROOT/'docs'/('CONFIRMATION_PROTOCOL_V1.json' if args.freeze else 'CONFIRMATION_PROTOCOL_V1_DRAFT.json')
    if args.freeze:
        if destination.exists():raise FileExistsError(destination)
        for r in [1,2]:
            report=json.loads((ROOT/'reports'/f'training_repeat{r}_pipeline.json').read_text())
            if report['status']!='completed_development_training_replication':raise ValueError('Training replication incomplete')
        protocol['status']='frozen_before_reserved_evaluation'
    destination.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'path':str(destination),'status':protocol['status'],'sha256':sha256(destination)}))


if __name__=='__main__':main()
