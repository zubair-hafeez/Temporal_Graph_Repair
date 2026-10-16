"""Figures from complete declared trial groups, with report/source hashes."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256

LABELS={'bpi2017':'BPI 2017\nApplications / offers',
        'bpi2019':'BPI 2019\nPurchase orders / items',
        'bpi2012_check':'BPI 2012 check\nApplications (denormalized)',
        'forgeflow':'ForgeFlow\nOrders / shipments','permitflow':'PermitFlow\nApplications / inspections',
        'procurechange':'ProcureChange\nRequisitions / invoices','trialversion':'TrialVersion\nParticipants / samples'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',required=True)
    parser.add_argument('--domains',nargs='+',required=True);parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--primary-only',action='store_true',help='Plot the paired training-data comparison; controls remain in tables')
    args=parser.parse_args();path=ROOT/'reports'/f'{args.evidence}.json';evidence=json.loads(path.read_text())
    for name,digest in evidence['source_reports'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Summarized report changed: '+name)
    groups={(r['domain'],r['suite'],r['arm']):r for r in evidence['groups']}
    def metric(domain,suite,arm,field):
        row=groups[domain,suite,arm]
        if row['completed_trials']!=row['expected_trials'] or row['metrics'] is None:
            raise ValueError('Required trial group incomplete: '+str((domain,suite,arm)))
        value=row['metrics'][field]
        if value['defined_trials']!=row['expected_trials']:
            raise ValueError('Figure requires every trial to define its plotted metric')
        return value
    arms=[('Damaged input','primary','varied','before_f1','#949da5'),
          ('Original observations','primary','plain','after_f1','#cf946b'),
          ('Varied observations','primary','varied','after_f1','#287f91'),
          ('Source trained (256 + calibration)','source','source256','after_f1','#8165a5'),
          ('Target-local extra corruption','adaptation','target_selfsupervised','after_f1','#7c994e')]
    if args.primary_only:arms=arms[:3]
    # Validate everything before writing any output.
    plotted={(d,s,a,f):metric(d,s,a,f) for d in args.domains for _,s,a,f,_ in arms}
    for d in args.domains:
        for suite,field in [('pipeline','gain_f1'),('clean','newly_damaged_events')]:
            plotted[d,suite,'varied',field]=metric(d,suite,'varied',field)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(max(8,len(args.domains)*3),4.7))
    centers=np.arange(len(args.domains));width=.7/len(arms)
    for j,(label,suite,arm,field,color) in enumerate(arms):
        values=[plotted[d,suite,arm,field] for d in args.domains]
        means=np.array([r['mean'] for r in values]);low=np.array([r['min'] for r in values]);high=np.array([r['max'] for r in values])
        ax.bar(centers+(j-(len(arms)-1)/2)*width,means,width=width,color=color,label=label,
               yerr=np.vstack([means-low,high-means]),capsize=2,error_kw={'elinewidth':.9})
    ax.set_xticks(centers,[LABELS.get(d,d) for d in args.domains]);ax.set_ylim(0,1.02)
    ax.set_ylabel('Entity-history F1');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.27),frameon=False,fontsize=9)
    fig.subplots_adjust(left=.09,right=.99,bottom=.21,top=.76)
    output=ROOT/'paper/figures';prefix=args.output_prefix
    for suffix in ['pdf','png']:fig.savefig(output/(prefix+'_history.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(max(10,len(args.domains)*3.4),4.5))
    for ax,suite,field,title,ylabel,color in [
        (axes[0],'pipeline','gain_f1','Unseen recording-fault mechanisms','History F1 change from damaged input','#287f91'),
        (axes[1],'clean','newly_damaged_events','Unnecessary changes on clean inputs','Initially correct events damaged','#b6674d')]:
        values=[plotted[d,suite,'varied',field] for d in args.domains]
        means=np.array([r['mean'] for r in values]);low=np.array([r['min'] for r in values]);high=np.array([r['max'] for r in values])
        ax.bar(centers,means,width=.5,color=color,yerr=np.vstack([means-low,high-means]),capsize=3)
        if suite=='pipeline':ax.axhline(0,color='#4e5d66',linewidth=1)
        else:ax.set_ylim(bottom=0)
        ax.set_xticks(centers,[LABELS.get(d,d).split('\n')[0] for d in args.domains],rotation=20,ha='right')
        ax.set_title(title);ax.set_ylabel(ylabel);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.tight_layout(w_pad=3)
    for suffix in ['pdf','png']:fig.savefig(output/(prefix+'_failure_boundaries.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    source={'evidence_report':str(path.relative_to(ROOT)),'evidence_sha256':sha256(path),
            'script_sha256':sha256(__file__),'domains':args.domains,'primary_only':args.primary_only,
            'plotted_metrics':[{'domain':k[0],'suite':k[1],'arm':k[2],'field':k[3],**v} for k,v in plotted.items()],
            'error_bars':'Observed minimum and maximum across the three declared joint trials, not confidence intervals.'}
    (output/(prefix+'_sources.json')).write_text(json.dumps(source,indent=2)+'\n')


if __name__=='__main__':main()
