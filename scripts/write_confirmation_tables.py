"""Generate paper tables from complete, provenance-checked trial groups."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256

NAMES={'bpi2017':'BPI 2017','bpi2019':'BPI 2019','bpi2012_check':'BPI 2012 check',
       'forgeflow':'ForgeFlow','permitflow':'PermitFlow','procurechange':'ProcureChange','trialversion':'TrialVersion'}


def cell(metric):
    if metric['defined_trials']!=len(metric['values']):raise ValueError('Undefined table metric')
    return r'\shortstack{'+f"{metric['mean']:.3f}"+r'\\{\scriptsize ['+f"{metric['min']:.3f}, {metric['max']:.3f}"+']}}'


def table(headers, rows, caption, label):
    return '\n'.join([r'\begin{table}[ht]',r'\centering\small',
        r'\resizebox{\linewidth}{!}{\begin{tabular}{l'+'c'*(len(headers)-1)+'}',r'\toprule',
        ' & '.join(headers)+r'\\',r'\midrule',*[' & '.join(row)+r'\\' for row in rows],
        r'\bottomrule',r'\end{tabular}}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table}',''])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',required=True)
    parser.add_argument('--domains',nargs='+',required=True);parser.add_argument('--output-prefix',required=True)
    args=parser.parse_args();source=ROOT/'reports'/f'{args.evidence}.json';report=json.loads(source.read_text())
    for name,digest in report['source_reports'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('A summarized source report changed: '+name)
    groups={(g['domain'],g['suite'],g['arm']):g for g in report['groups']}
    def metrics(domain,suite,arm):
        group=groups[domain,suite,arm]
        if group['metrics'] is None or group['completed_trials']!=group['expected_trials']:
            raise ValueError('Table group incomplete: '+str((domain,suite,arm)))
        return group['metrics']
    main_rows=[];safety_rows=[];query_rows=[];type_rows=[]
    for domain in args.domains:
        a=metrics(domain,'primary','plain');b=metrics(domain,'primary','varied')
        s=metrics(domain,'source','source256');t=metrics(domain,'adaptation','target_selfsupervised')
        main_rows.append([NAMES[domain],cell(b['before_f1']),cell(a['after_f1']),cell(b['after_f1']),cell(s['after_f1']),cell(t['after_f1'])])
        clean=metrics(domain,'clean','varied');pipeline=metrics(domain,'pipeline','varied')
        safety_rows.append([NAMES[domain],cell(b['before_exact']),cell(b['after_exact']),
            '/'.join(str(int(v)) for v in b['identity_false_pairs']['values']),
            '/'.join(str(int(v)) for v in b['newly_damaged_events']['values']),
            '/'.join(str(int(v)) for v in clean['newly_damaged_events']['values']),cell(pipeline['gain_f1'])])
        for title,key in [('First recorded time','first_timestamp_correct_fraction'),
                          ('Last recorded time','last_timestamp_correct_fraction'),
                          ('Cancellation evidence','cancellation_evidence_f1')]:
            before=b['before_'+key];after=b['after_'+key]
            if before['defined_trials']==3 and after['defined_trials']==3:
                query_rows.append([NAMES[domain],title,cell(before),cell(after)])
        types=sorted({r['type'] for r in report['per_type_results'] if r['domain']==domain and r['suite']=='primary'})
        for typ in types:
            selected=sorted([r for r in report['per_type_results'] if r['domain']==domain and r['suite']=='primary' and r['type']==typ and r['arm'].startswith('varied_')],key=lambda r:r['seed'])
            if len(selected)!=3:raise ValueError('Incomplete type result')
            def type_cell(key):
                values=[r[key] for r in selected]
                return cell({'values':values,'defined_trials':3,'mean':sum(values)/3,'min':min(values),'max':max(values)})
            type_rows.append([NAMES[domain],typ,str(selected[0]['entities']),type_cell('before_f1'),type_cell('after_f1')])
    ablation_rows=[]
    variants=[('Varied observations, full','primary','varied_r0'),
        ('Without temporal scoring','controls','no_temporal_scoring'),
        ('Without context scoring','controls','no_context_scoring'),
        ('One pass','controls','one_pass'),('Fixed sequence','controls','fixed_sequence'),
        ('Original + stable-field view','controls','stable_fields'),
        ('GraphSAGE edits','controls','sage'),('MLP edits','controls','mlp'),
        ('Temporal-message edits','controls','temporal')]
    for title,suite,arm in variants:
        row=[title]
        for domain in args.domains:
            selected=[r for r in report['trial_results'] if r['domain']==domain and r['suite']==suite and r['arm']==arm]
            if len(selected)!=1:raise ValueError('Missing or ambiguous ablation row: '+str((domain,suite,arm)))
            row.append(f"{selected[0]['after_f1']:.4f}")
        ablation_rows.append(row)
    text={
      'history':table(['Source','Damaged','Original','Varied','Source labels','Target-local'],main_rows,
        'Frozen confirmation: entity-history F1, mean and observed range over three trials. Original and varied training share executed worlds and generated calibration within a trial. Source supervision uses 256 verified BPI 2012 training histories plus additional source calibration labels. Target-local learning uses extra corruption of the imperfect target and generated calibration. BPI 2012 check retains explicitly denormalized requested amount.',
        'tab:confirmation-history'),
      'safety':table(['Source','Exact before','Exact after','False merge pairs','Mixed: new damage','Clean: new damage','Pipeline F1 change'],safety_rows,
        'Recovery and failure boundaries of the varied-observation learner. Exact-history fractions and pipeline changes show trial means/ranges. Counts separated by slashes give the three trial values in order. New damage counts initially correct events whose target participation becomes wrong. False merge pairs are inferred identity pairs, not merely the number of merge operations. Clean inputs are uncorrupted source references, not certified error-free organizational records.',
        'tab:confirmation-safety'),
      'controls':table(['Control',*[NAMES[d] for d in args.domains]],ablation_rows,
        'Single-trial controls on the same native (or declared denormalized) mixed-fault inputs, seed 91001. Refitted scoring ablations retain temporal/contextual candidate retrieval. Neural controls share the candidate universe and training collections, with the optimization budgets described in the appendix. Their scores are adaptations to this edit task, not native reproductions of published trace-repair methods.',
        'tab:confirmation-controls'),
      'queries':table(['Source','Recorded-evidence query','Before','After'],query_rows,
        'Operational queries for the varied-observation learner under mixed faults, with trial means and observed ranges. Boundary-time scores concern timestamps, not every simultaneous event. Cancellation evidence is an event-retrieval query, not a causal or business-outcome prediction. Undefined query F1 is not replaced by zero.',
        'tab:confirmation-queries'),
      'types':table(['Source','Target type','Entities','Before','After'],type_rows,
        'History F1 by target entity type for varied-observation training, with three-trial means and observed ranges. Each source cohort is fixed across fault trials. These views expose differences obscured by pooling multiple entity types.',
        'tab:confirmation-types')}
    output=ROOT/'paper/tables';output.mkdir(exist_ok=True)
    for name,content in text.items():(output/(args.output_prefix+'_'+name+'.tex')).write_text(content)
    (output/(args.output_prefix+'_sources.json')).write_text(json.dumps({
        'report':str(source.relative_to(ROOT)),'report_sha256':sha256(source),
        'script_sha256':sha256(__file__),'domains':args.domains,'files':list(text)},indent=2)+'\n')


if __name__=='__main__':main()
