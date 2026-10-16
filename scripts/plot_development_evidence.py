"""Export development-only figures from measured reports, without fitted curves."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tgrepair.data import sha256


def main():
    out=ROOT/'paper/figures';out.mkdir(parents=True,exist_ok=True)
    sources={}
    def read(name):
        path=ROOT/'reports'/f'{name}.json';sources[name]=sha256(path)
        return json.loads(path.read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.7),constrained_layout=True)
    for ax,domain in zip(axes,['battery','claim']):
        native=read(f'external_{domain}_mixed_v1');structural=read(f'external_{domain}_structural_v1')
        calibrated=read(f'external_{domain}_calibration_only_v3');trained=read(f'external_{domain}_attribute_training_v3')
        values=[native['before']['f1'],native['after']['history']['f1'],structural['after']['history']['f1'],
                calibrated['after']['history']['f1'],trained['after']['history']['f1']]
        bars=ax.bar(['Damaged','Original\ntraining','Attributes\nhidden','Calibration\nonly','Varied-attribute\ntraining'],values,
                    color=['#9aa3ad','#c66547','#8db5c8','#c7aa73','#367a9b'],width=.65)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.004,f'{value:.3f}',ha='center',fontsize=10)
        ax.set_ylim(0,1.);ax.set_ylabel('Mean entity-history F1');ax.tick_params(axis='x',labelsize=8)
        ax.set_title('Battery service' if domain=='battery' else 'Insurance claims')
    fig.savefig(out/'development_observation_views.pdf');fig.savefig(out/'development_observation_views.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,3.7),constrained_layout=True)
    budgets=[16,64,256];positions=np.arange(3)
    for offset,regime,label,color in [(-.18,'source','Source histories','#9aa3ad'),(.18,'mixed','Source + generated histories','#367a9b')]:
        values=[read(f'source_decode_{regime}{budget}_v3')['after']['history']['f1'] for budget in budgets]
        bars=ax.bar(positions+offset,values,width=.34,label=label,color=color)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.003,f'{value:.3f}',ha='center',fontsize=9)
    ax.set_xticks(positions,[str(b) for b in budgets]);ax.set_xlabel('Verified training histories (additional calibration labels used)')
    ax.set_ylabel('Mean entity-history F1');ax.set_ylim(0,1.12);ax.legend(frameon=False,loc='upper left')
    fig.savefig(out/'development_annotation_budgets.pdf');fig.savefig(out/'development_annotation_budgets.png',dpi=160);plt.close(fig)
    (out/'development_sources.json').write_text(json.dumps({'status':'development_only_single_run_figures',
                                                         'source_report_hashes':sources,'plot_script_hash':sha256(__file__)},indent=2)+'\n')


if __name__=='__main__':main()
