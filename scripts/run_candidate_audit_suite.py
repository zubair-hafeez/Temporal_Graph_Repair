"""Initial candidate coverage for every declared fault trial of selected sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--domains',nargs='+',required=True)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError('Use one to four audit workers')
    protocol_path=ROOT/'docs/CONFIRMATION_PROTOCOL_V1.json';protocol=json.loads(protocol_path.read_text())
    planned=[]
    for domain in args.domains:
        view=next(v for v in protocol['sources'][domain]['views'] if v!='structural')
        for variant in protocol['fault_families']:
            for seed in protocol['fault_seeds']:
                task=Path(protocol['task_root'])/domain/view/f'{variant}_{seed}'
                planned.append({'domain':domain,'view':view,'variant':variant,'seed':seed,'task':str(task),
                                'output':f'candidate_{domain}_{view}_{variant}_{seed}'})
    report={'status':'running_candidate_audit_suite','protocol_hash':sha256(protocol_path),'jobs':planned,'outcomes':[]}
    destination=ROOT/'reports'/f'{args.output}.json'
    def run(row):
        task_path=ROOT/row['task']/'task.json';task=json.loads(task_path.read_text())
        if task['protocol_hash']!=report['protocol_hash']:raise ValueError('Task belongs to another protocol')
        output=ROOT/'reports'/f"{row['output']}.json"
        if not output.exists():
            command=[sys.executable,str(ROOT/'scripts/audit_task_candidates.py'),'--task',row['task'],'--output',row['output']]
            with output.with_suffix('.log').open('a') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads(output.read_text())
        if result['task_hash']!=sha256(task_path) or result['status']!='evaluator_only_initial_candidate_audit':
            raise ValueError('Existing audit does not match current task')
        return {'output':row['output'],'status':'completed','hash':sha256(output)}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending={executor.submit(run,row):row for row in planned}
        for future in as_completed(pending):
            try: result=future.result()
            except Exception as error: result={'output':pending[future]['output'],'status':'failed','error':repr(error)}
            report['outcomes'].append(result);destination.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    report['status']='completed_candidate_audit_suite' if all(r['status']=='completed' for r in report['outcomes']) else 'candidate_audit_suite_has_failures'
    destination.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
