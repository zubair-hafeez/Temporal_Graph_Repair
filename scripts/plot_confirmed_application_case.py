"""Show changed/cancellation evidence from the declared loan case audit."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256


def main():
    source=ROOT/'reports/confirm_bpi2017_application_cases_v1.json'
    report=json.loads(source.read_text());cases=report['cases'];shown=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11.3,4.4),gridspec_kw={'width_ratios':[1.8,1]},constrained_layout=True)
    for ax,case in zip(axes,cases):
        events=case['events']
        if len(events)>10:
            events=[e for e in events if e['retrieved_before']!=e['retrieved_after'] or 'cancel' in e['activity'].lower()]
        shown.append({'entity':case['true'],'event_ids':[e['uid'] for e in events],
                      'shown_records':len(events),'full_reference_records':case['records']})
        start=min(e['time'] for e in case['events']);labels=[]
        for e in events:
            activity=e['activity'].split('::')[0].split('_',1)[-1]
            if activity=='Call incomplete files':activity='Call about\nincomplete files'
            if activity=='Cancelled':activity=('Application canceled' if e['activity'].startswith('A_') else 'Offer canceled')
            if not e['in_reference_history']:activity+='\n(other entity)'
            labels.append(activity+'\n'+f"+{(e['time']-start)/3600:.1f} h")
        for y,key in [(2,'in_reference_history'),(1,'retrieved_before'),(0,'retrieved_after')]:
            ax.axhline(y,color='#e0e4e7',lw=1,zorder=0)
            for i,e in enumerate(events):
                color='#a45139' if not e['in_reference_history'] else '#337b8d'
                if e['in_reference_history'] and 'cancel' in e['activity'].lower():color='#7856a4'
                present=e[key]
                ax.scatter(i,y,s=130,facecolors=color if present else 'white',edgecolors=color if present else '#b8c0c7',linewidths=1.3,zorder=2)
        ax.set_yticks([2,1,0],['Reference','Before repair','After repair'])
        ax.set_xticks(range(len(events)),labels,rotation=32,ha='right',fontsize=8)
        ax.set_ylim(-.5,2.6);ax.set_xlim(-.5,len(events)-.5)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(length=0)
        title='Application: offer record recovered' if case['gain']>0 else 'Offer: short history damaged'
        ax.set_title(title+'\n'+f"Full-history F1 {case['before_f1']:.3f} → {case['after_f1']:.3f}",fontsize=11)
    output=ROOT/'paper/figures'
    for suffix in ['pdf','png']:fig.savefig(output/('confirmation_loan_cases.'+suffix),dpi=180)
    plt.close(fig)
    (output/'confirmation_loan_cases_sources.json').write_text(json.dumps({
        'audit_report':str(source.relative_to(ROOT)),'audit_sha256':sha256(source),'plot_script_sha256':sha256(__file__),
        'display_rule':'For a case with more than ten displayed-union records, show every changed association and every cancellation record; otherwise show all records.',
        'shown':shown,'limits':'Post hoc illustration from primary replicate zero. Scores concern full histories, including records not shown. Columns are chronological but evenly spaced.'},indent=2)+'\n')


if __name__=='__main__':main()
