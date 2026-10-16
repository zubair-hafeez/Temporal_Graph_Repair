"""Archive a terminated run with zero completed rounds and replay its config."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    report_path=ROOT/'reports'/f'{args.run}.json';report=json.loads(report_path.read_text())
    if report['status']!='running_confirmation_evaluation':raise ValueError('Run is not an unfinished confirmation')
    folder=ROOT/'data/decoding'/args.run
    if not folder.exists() or list(folder.iterdir()):raise ValueError('Only an empty, zero-round output can be restarted here')
    config=report['config']
    if config['output']!=args.run or not config['protocol']:raise ValueError('Run configuration mismatch')
    if report['pins']['protocol_hash']!=sha256(ROOT/config['protocol']):raise ValueError('Protocol changed')
    archive=ROOT/'reports/interrupted'/f'{args.run}_before_first_round_{sha256(report_path)[:16]}.json'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():raise FileExistsError(archive)
    report_path.rename(archive)
    empty_archive=ROOT/'data/decoding/interrupted'/archive.stem
    empty_archive.parent.mkdir(parents=True,exist_ok=True)
    folder.rename(empty_archive)
    command=[sys.executable,str(ROOT/'scripts/evaluate_external_policy.py')]
    allowed={'task','model','kind','output','rounds','participant_budget','pair_budget','precision',
             'calibration','attribute_view','stability_threshold','protocol','mode'}
    for key,value in config.items():
        if key=='resume':continue
        if key not in allowed:raise ValueError('Unknown configuration field: '+key)
        if value is not None:command.extend(['--'+key.replace('_','-'),str(value)])
    record={'status':'restarting_before_first_completed_round','original_report':str(archive.relative_to(ROOT)),
            'original_report_hash':sha256(archive),'script_hash':sha256(__file__),'command':command,
            'reason':'Interrupted before the first round; the existing resume path requires a checkpoint. No decoded result is discarded.'}
    record_path=ROOT/'reports'/f'{args.run}_operational_restart.json'
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    with (ROOT/'reports'/f'{args.run}.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    record['status']='completed_zero_round_restart';record['completed_report_hash']=sha256(report_path)
    record_path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
