"""Export measured history examples, including a failure, for the manuscript."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256


def main():
    source=ROOT/'reports/battery_application_cases_v1.json'
    report=json.loads(source.read_text());cases=report['cases']
    out=ROOT/'paper/figures'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.9),gridspec_kw={'width_ratios':[1.8,1]},constrained_layout=True)
    for ax,case in zip(axes,cases):
        events=case['events'];x=np.arange(len(events));first=min(e['time'] for e in events)
        labels=[]
        for e in events:
            activity=e['activity'].replace('State of Health Assessed','Health assessed').replace('Warranty Claim','Warranty claim').replace('Service Order','Service order')
            if not e['in_reference_history']:activity+='\n(other order)'
            labels.append(activity+'\n'+f"+{(e['time']-first)/3600:.1f} h")
        for y,key in [(2,'in_reference_history'),(1,'retrieved_before'),(0,'retrieved_after')]:
            ax.axhline(y,color='#e0e4e7',lw=1,zorder=0)
            for i,e in enumerate(events):
                present=e[key]
                color=('#a45139' if not e['in_reference_history'] else '#337b8d')
                if e['attributes'].get('is_exception') in {1,True,'1'}:color='#7856a4'
                ax.scatter(i,y,s=135,marker='o',facecolors=color if present else 'white',
                           edgecolors=color if present else '#b8c0c7',linewidths=1.3,zorder=2)
        ax.set_yticks([2,1,0],['Reference','Before repair','After repair'])
        ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=8)
        ax.set_ylim(-.5,2.6);ax.set_xlim(-.5,len(events)-.5)
        ax.spines['left'].set_visible(False);ax.spines['bottom'].set_visible(False);ax.tick_params(length=0)
        title='Recovered warranty record' if case['gain']>0 else 'Short-history failure'
        ax.set_title(title+'\n'+f"History F1 {case['before_f1']:.3f} → {case['after_f1']:.3f}",fontsize=11)
    fig.savefig(out/'development_application_cases.pdf')
    fig.savefig(out/'development_application_cases.png',dpi=170)
    plt.close(fig)
    (out/'development_application_cases_sources.json').write_text(json.dumps({
        'status':'posthoc_development_illustration','audit_hash':sha256(source),'plot_script_hash':sha256(__file__),
        'legend':'Filled marker = retrieved record; empty marker = absent. Purple = recorded genuine exception. Rust = reference says a different order.'},indent=2)+'\n')


if __name__=='__main__':main()
