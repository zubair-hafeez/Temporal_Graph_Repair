"""Execute a predeclared confirmation block with bounded CPU concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.confirmation_contract import adaptation_names
from tgrepair.data import sha256


def jobs(protocol,domains,suite):
    rows=[]
    for domain in domains:
        native=next(v for v in protocol['sources'][domain]['views'] if v!='structural')
        for pair in protocol['primary_comparison']['pairs']:
            r=pair['replicate'];seed=pair['fault_seed']
            if suite=='primary':
                rows.extend([dict(domain=domain,view=native,variant='mixed',seed=seed,arm=f'{arm}_r{r}',model=pair[key])
                             for arm,key in [('plain','original'),('varied','varied')]])
            elif suite=='clean':rows.append(dict(domain=domain,view=native,variant='clean',seed=0,arm=f'varied_r{r}',model=pair['varied']))
            elif suite=='pipeline':rows.append(dict(domain=domain,view=native,variant='pipeline',seed=seed,arm=f'varied_r{r}',model=pair['varied']))
            elif suite=='source':rows.append(dict(domain=domain,view=native,variant='mixed',seed=seed,arm='source256',model='source_edit_learning_v2_source_only_256'))
            elif suite=='adaptation':rows.append(dict(domain=domain,view=native,variant='mixed',seed=seed,arm='target_selfsupervised',model=None))
        if suite=='structural':
            rows.extend([dict(domain=domain,view='structural',variant='mixed',seed=91001,arm=arm,model=model)
                         for arm,model in [('plain','edit_feature_v2'),('varied','edit_feature_attributes_v3')]])
        if suite=='controls':
            controls=[('no_temporal_scoring','edit_feature_no_temporal_v3',{}),('no_context_scoring','edit_feature_no_context_v3',{}),
                      ('one_pass','edit_feature_attributes_v3',{'mode':'one_pass'}),
                      ('fixed_sequence','edit_feature_attributes_v3',{'mode':'sequential'}),
                      ('stable_fields','edit_feature_v2',{'attribute_view':'stable','calibration':'edit_feature_v2_stable_calibration_v3'}),
                      *[(name,f'edit_{name}_attributes_v3',{}) for name in ['sage','mlp','temporal']]]
            rows.extend([dict(domain=domain,view=native,variant='mixed',seed=91001,arm=arm,model=model,**options)
                         for arm,model,options in controls])
    for row in rows:
        row['output']=f"confirm_{row['domain']}_{row['view']}_{row['variant']}_{row['seed']}_{row['arm']}"
        row['task']=str(Path(protocol['task_root'])/row['domain']/row['view']/f"{row['variant']}_{row['seed']}")
    return rows


def run_job(row,protocol,protocol_path,resume):
    task_dir=ROOT/row['task']
    if not (task_dir/'task.json').exists():raise FileNotFoundError('Task not prepared: '+str(task_dir))
    task=json.loads((task_dir/'task.json').read_text())
    if task['protocol_hash']!=sha256(protocol_path):raise ValueError('Task belongs to a different protocol')
    def execute(script,arguments,label):
        command=[sys.executable,str(ROOT/'scripts'/script),*arguments]
        with (ROOT/'reports'/f'{label}.log').open('a' if resume else 'x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        return command
    commands=[];model=row['model'];calibration=row.get('calibration')
    if model is None:
        names=adaptation_names(task)
        validation=protocol['target_local_adaptation']['validation_by_fault_seed'][str(row['seed'])]['name']
        stages=[('prepare_target_selfsupervision.py',['--task',row['task'],'--output',names['cache']],names['cache']),
                ('train_selfsupervised_controls.py',['--source-cache',names['cache'],'--regime','self_supervised','--output',names['training']],names['training']),
                ('score_edit_feature_validation.py',['--model',names['model'],'--validation',validation,'--output',names['calibration']],names['calibration'])]
        for script,arguments,label in stages:
            if not (ROOT/'reports'/f'{label}.json').exists():commands.append(execute(script,arguments,label))
            elif not resume:raise FileExistsError('Adaptation stage already exists; use --resume')
        model=names['model'];calibration=names['calibration'];kind='feature'
    else:
        spec=protocol['models'][model];kind=spec['kind']
        if calibration is None:calibration=model if model in spec['calibrations'] else next(iter(spec['calibrations']))
    result_path=ROOT/'reports'/f"{row['output']}.json"
    arguments=['--task',row['task'],'--model',model,'--kind',kind,'--calibration',calibration,
               '--protocol',str(protocol_path.relative_to(ROOT)),'--output',row['output'],
               '--mode',row.get('mode','joint'),'--attribute-view',row.get('attribute_view','all')]
    if result_path.exists():
        if not resume:raise FileExistsError(result_path)
        result=json.loads(result_path.read_text())
        if result['status']=='completed_confirmation_evaluation':
            if result['pins']['protocol_hash']!=sha256(protocol_path):raise ValueError('Completed result protocol mismatch')
            return {'output':row['output'],'status':'already_completed','report_hash':sha256(result_path)}
        arguments.append('--resume')
    start=time.monotonic();commands.append(execute('evaluate_external_policy.py',arguments,row['output']))
    result=json.loads(result_path.read_text())
    if result['status']!='completed_confirmation_evaluation':raise RuntimeError('Evaluation did not finish')
    return {'output':row['output'],'status':'completed','report_hash':sha256(result_path),
            'seconds':time.monotonic()-start,'commands':commands,'f1':result['after']['history']['f1']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='docs/CONFIRMATION_PROTOCOL_V1.json')
    parser.add_argument('--domains',nargs='+',required=True)
    parser.add_argument('--suite',choices=['primary','clean','pipeline','source','adaptation','structural','controls'],required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();protocol_path=ROOT/args.protocol;protocol=json.loads(protocol_path.read_text())
    rows=jobs(protocol,args.domains,args.suite)
    if args.plan_only:
        print(json.dumps({'status':protocol['status'],'count':len(rows),'jobs':rows},indent=2));return
    if protocol['status']!='frozen_before_reserved_evaluation':raise ValueError('Protocol not frozen')
    if not 1<=args.workers<=6:raise ValueError('Bounded concurrency requires 1..6 workers')
    name='confirmation_'+args.suite+'_'+'_'.join(args.domains)
    output=ROOT/'reports'/f'{name}.json';lock=ROOT/'data'/f'.{name}.lock'
    with lock.open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists() and not args.resume:raise FileExistsError(output)
        report={'status':'running_confirmation_block','config':vars(args),'protocol_hash':sha256(protocol_path),
                'jobs':rows,'outcomes':[]}
        output.write_text(json.dumps(report,indent=2)+'\n')
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending={executor.submit(run_job,row,protocol,protocol_path,args.resume):row for row in rows}
            for future in as_completed(pending):
                try:result=future.result()
                except Exception as error:result={'output':pending[future]['output'],'status':'failed','error':repr(error)}
                report['outcomes'].append(result)
                output.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(result),flush=True)
        report['status']='completed_confirmation_block' if all(r['status']!='failed' for r in report['outcomes']) else 'confirmation_block_has_failures'
        output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
