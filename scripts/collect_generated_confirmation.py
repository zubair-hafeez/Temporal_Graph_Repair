"""Aggregate all six frozen generated tests without event-level pseudo-replication."""
import argparse
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    protocol_path=ROOT/'docs/CONFIRMATION_PROTOCOL_V1.json';protocol=json.loads(protocol_path.read_text())
    protocol_hash=sha256(protocol_path);count=protocol['generated_confirmation']['count']
    sources={};trials=[];baseline=None;ordered=None
    for pair in protocol['primary_comparison']['pairs']:
        r=pair['replicate'];values={}
        for arm,key in [('plain','original'),('varied','varied')]:
            name=f'confirm_generated_{arm}_r{r}';path=ROOT/'reports'/f'{name}.json'
            report=json.loads(path.read_text())
            if report['status']!='completed_generated_confirmation':raise ValueError('Incomplete generated trial: '+name)
            if report['protocol_hash']!=protocol_hash or report['config']['model']!=pair[key]:raise ValueError('Protocol/model mismatch')
            if len(report['results'])!=2*count:raise ValueError('Wrong number of graph variants')
            current=[(row['episode'],row['group'],row['variant']) for row in report['results']]
            if ordered is None:ordered=current
            elif ordered!=current:raise ValueError('Generated trials evaluate different ordered worlds')
            bases=[row['before']['f1'] for row in report['results']]
            if baseline is None:baseline=bases
            elif baseline!=bases:raise ValueError('Generated initial outcomes differ')
            arm_values={}
            for variant in ['mixed','clean']:
                rows=[row for row in report['results'] if row['variant']==variant]
                if len(rows)!=count:raise ValueError('Incorrect variant count')
                mean_after=statistics.mean(row['after']['history']['f1'] for row in rows)
                if abs(mean_after-report['summary'][variant]['mean_after_f1'])>1e-12:raise ValueError('Summary inconsistent with graph outcomes')
                arm_values[variant]={'before_f1':statistics.mean(row['before']['f1'] for row in rows),
                    'after_f1':mean_after,'before_exact':statistics.mean(row['before']['exact_history_fraction'] for row in rows),
                    'after_exact':statistics.mean(row['after']['history']['exact_history_fraction'] for row in rows),
                    'improved_worlds':sum(row['after']['history']['f1']>row['before']['f1'] for row in rows),
                    'worsened_worlds':sum(row['after']['history']['f1']<row['before']['f1'] for row in rows),
                    'edits':sum(row['edit_count'] for row in rows)}
            values[arm]=arm_values;sources[str(path.relative_to(ROOT))]=sha256(path)
        trials.append({'replicate':r,**values,'paired_mixed_f1_gain':values['varied']['mixed']['after_f1']-values['plain']['mixed']['after_f1']})
    summaries={}
    for arm in ['plain','varied']:
        summaries[arm]={}
        for variant in ['mixed','clean']:
            summaries[arm][variant]={}
            for metric in trials[0][arm][variant]:
                values=[row[arm][variant][metric] for row in trials]
                summaries[arm][variant][metric]={'values':values,'mean':statistics.mean(values),'min':min(values),'max':max(values)}
    report={'status':'complete_generated_confirmation_analysis','protocol_hash':protocol_hash,'source_reports':sources,
            'script_hash':sha256(__file__),'worlds_per_model':count,'trials':trials,'summaries':summaries,
            'limits':['Each trial is a separately generated training/validation collection; all models evaluate the same held-out worlds.',
                      'World-level performance is first averaged equally; training trials, not events, define the reported range.',
                      'All worlds share a simulator. This evaluates held-out compositions of known mechanisms, not independent-domain transfer.']}
    output=ROOT/'reports'/f'{args.output}.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'mixed':{a:summaries[a]['mixed']['after_f1'] for a in summaries},
                      'clean':{a:summaries[a]['clean']['after_f1'] for a in summaries}}),flush=True)


if __name__=='__main__':main()
