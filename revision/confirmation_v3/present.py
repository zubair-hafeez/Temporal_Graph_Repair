"""Evidence-linked tables, plots and readable complete confirmation report."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tgrepair.data import sha256
from .run import HERE, DOMAINS, ARMS, write


def main():
    a=json.loads((HERE/'analysis.json').read_text()); assert a['conditions']==768
    rows=a['aggregates']; claims=[]
    def get(d,b,arm,q,policy='guarded'):
        return next(r for r in rows if (r['domain'],r['budget'],r['arm'],r['precision'],r['policy'])==(d,b,arm,q,policy))
    def val(r,k,scale=1.,digits=1):
        value=r[k]['mean']*scale; literal=f'{value:.{digits}f}'
        claims.append({'aggregate_index':rows.index(r),'field':k,'scale':scale,'digits':digits,'value':value,'literal':literal})
        return literal
    def tex_table(caption,label,columns,header,body):
        return '\n'.join([r'\begin{table}[t]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
            r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',r'\toprule',
            header+r' \\ \midrule',*body,r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n'
    body=[]; lines=['FRESH-HISTORY CONFIRMATION: COMPLETE RESULTS','',
        '768 conditions, 36 unchanged fitted models, 8 disjoint 128-root blocks per source.',
        'Means below average 24 block/model-seed runs per setting. R=full-event recovery;',
        'H=newly damaged originally correct events (full identity-cluster expansion).',
        'F1 is percent. Counts are per 128-root block, not per 1,024-root source.',
        'All errors are injected; target histories are fresh, source processes are familiar.','',
        'PRIMARY GRID: source budget q | Real8 F1 R H | +Generated F1 R H']
    for d,label in zip(DOMAINS,['Loans','Purchasing']):
        for b in [16,64,256]:
            for q in [.95,.99]:
                r,g=[get(d,b,arm,q) for arm in ARMS]
                cells=[label if (b,q)==(16,.95) else '',str(b),f'{q:.2f}']
                for k,scale in [('f1',100),('recovery',1),('damage',1)]: cells += [val(x,k,scale,2 if k=='damage' else 1) for x in [r,g]]
                body.append(' & '.join(cells)+r' \\')
                lines.append(f"{d} {b} {q} | "+' | '.join(' '.join(f'{x[k]["mean"]*scale:.2f}' for k,scale in [('f1',100),('recovery',1),('damage',1)]) for x in [r,g]))
        if d==DOMAINS[0]: body.append(r'\midrule')
    caption=r'Untouched-history confirmation: means over eight disjoint 128-root blocks and three fixed fitted-model/fault seeds. $R_8$ repeatedly corrupts the same verified histories; $+G$ adds generated training. Budgets include calibration. $R$ counts fully recovered events and $H$ newly damaged events per block; all false-merge effects are included. History F1 is percent.'
    (HERE/'paper_table_primary.tex').write_text(tex_table(caption,'tab:fresh-primary','lrr rr rr rr',
        r'Source & Budget & $q$ & F1 $R_8$ & F1 $+G$ & $R$ $R_8$ & $R$ $+G$ & $H$ $R_8$ & $H$ $+G$',body))
    body=[]; lines += ['', 'SAFETY: budget 64, q=.99; mean per 128-root block','source arm policy | F1 R H truePairs falsePairs']
    for d,label in zip(DOMAINS,['Loans','Purchasing']):
        for arm,alabel in zip(ARMS,[r'$R_8$',r'$+G$']):
            for policy,plabel in [('unguarded','Unguarded'),('guarded','Guarded'),('never_merge','Never merge')]:
                r=get(d,64,arm,.99,policy)
                cells=[label if (arm,policy)==('real8','unguarded') else '',alabel if policy=='unguarded' else '',plabel]
                cells += [val(r,k,scale,2 if k in ['damage','correct_pairs','false_pairs'] else 1) for k,scale in [('f1',100),('recovery',1),('damage',1),('correct_pairs',1),('false_pairs',1)]]
                body.append(' & '.join(cells)+r' \\')
                lines.append(f'{d} {arm} {policy} | '+' '.join(f'{r[k]["mean"]*scale:.3f}' for k,scale in [('f1',100),('recovery',1),('damage',1),('correct_pairs',1),('false_pairs',1)]))
        if d==DOMAINS[0]: body.append(r'\midrule')
    (HERE/'paper_table_safety.tex').write_text(tex_table(
        r'Merge-policy confirmation at 64 verified bundles and $q=.99$. $R$ and $H$ are fully recovered and newly damaged events. TP/FP count true/false merged identity pairs. Never merge retains the guarded participation rules and atomic root replacements. Counts and F1 (percent) average the same 24 runs per setting.',
        'tab:fresh-safety','lll rrrrr',r'Source & Arm & Policy & F1 & $R$ & $H$ & TP & FP',body))
    lines += ['','PAIRED CONTRASTS: left minus right; mean [conditional paired block-bootstrap 95% interval]',
              'Three fixed model/fault runs are averaged within each block before 10,000 resamples.',
              'Intervals are conditional on these fitted models and source processes; no multiplicity adjustment.']
    hypotheses=[]
    for c in a['contrasts']:
        lines.append(f"{c['domain']} budget{c['budget']} q{c['precision']} {c['left']} - {c['right']}")
        for k in ['f1','recovery','damage','correct_pairs','false_pairs']:
            m=c['metrics'][k]; scale=100 if k=='f1' else 1
            lines.append(f"  {k}: {m['mean']*scale:.4f} [{m['ci95'][0]*scale:.4f}, {m['ci95'][1]*scale:.4f}], positive/negative blocks {m['positive_blocks']}/{m['negative_blocks']}")
        m=c['metrics']
        if c['domain']=='bpi2017' and c['budget']==64 and c['precision']==.99 and c['left']==['real8_generated','guarded'] and c['right']==['real8','guarded']:
            hypotheses.append({'hypothesis':'H1','mean_direction':m['recovery']['mean']>0 and m['damage']['mean']<=0,
                               'interval_supported':m['recovery']['ci95'][0]>0 and m['damage']['ci95'][1]<=0,'contrast':c})
        if c['right'][1]=='never_merge':
            retained=get(c['domain'],64,c['left'][0],.99)['correct_pairs']['mean']>0
            hypotheses.append({'hypothesis':'H2','domain':c['domain'],'arm':c['left'][0],
                'mean_direction':retained and (m['f1']['mean']>0 or m['recovery']['mean']>0) and m['damage']['mean']<=0,
                'interval_supported':retained and (m['f1']['ci95'][0]>0 or m['recovery']['ci95'][0]>0) and m['damage']['ci95'][1]<=0,'contrast':c})
    lines += ['','PRESPECIFIED HYPOTHESIS DECISIONS']
    for h in hypotheses: lines.append(str({k:v for k,v in h.items() if k!='contrast'}))
    contrast_body=[]; contrast_claims=[]
    for h in hypotheses:
        c=h['contrast']; label='Loans' if c['domain']=='bpi2017' else 'Purchasing'
        if h['hypothesis']=='H1': name=r'$+G-R_8$'
        else: name=(r'$R_8$' if h['arm']=='real8' else r'$+G$')+': guard--never'
        cells=[label,name]
        for k in ['f1','recovery','damage']:
            m=c['metrics'][k]; scale=100 if k=='f1' else 1; digits=1 if k=='recovery' else 2
            numbers=[m['mean'],*m['ci95']]
            formatted=[f'{v*scale:.{digits}f}' for v in numbers]
            cells.append(formatted[0]+' ['+', '.join(formatted[1:])+']')
            contrast_claims.append({'contrast_index':a['contrasts'].index(c),'field':k,'scale':scale,
                                    'digits':digits,'values':numbers,'literals':formatted})
        contrast_body.append(' & '.join(cells)+r' \\')
    (HERE/'paper_table_contrasts.tex').write_text(tex_table(
        r'Prespecified contrasts at 64 bundles and $q=.99$: mean differences [conditional 95\% intervals]. The first row tests augmentation with guarded decoding; other rows subtract never merge from guarded decoding. F1 differences are percentage points; recovery and harm are events per 128-root block. Intervals resample eight blocks after averaging three fixed model/fault runs; no multiplicity correction is applied.',
        'tab:fresh-contrasts','ll rrr',r'Source & Contrast & $\Delta$ F1 & $\Delta R$ & $\Delta H$',contrast_body).replace(r'\centering\small',r'\centering\footnotesize'))
    write(HERE/'contrast_claims.json',{'analysis_sha256':sha256(HERE/'analysis.json'),'claims':contrast_claims})
    seed_report=json.loads((HERE/'seed_sensitivity.json').read_text())
    lines += ['','FITTED-SEED SENSITIVITY OF H1: descriptive, conditional on the existing fits']
    for row in seed_report['seeds']:
        lines.append(f"seed{row['seed']}: recovery difference {row['difference']['recovery']:.3f}; damage difference {row['difference']['damage']:.3f}; F1 difference {100*row['difference']['f1']:.3f} percentage points")
    lines.append('One of three fitted-seed means loses recovery. Block-bootstrap intervals do not cover training-population uncertainty.')
    lines += ['','COHORT AND COMPUTE']
    for d in DOMAINS:
        manifest=json.loads((HERE/'data'/d/'cohort.json').read_text()); summary=json.loads((HERE/'results'/d/'summary.json').read_text())
        lines.append(f"{d}: {sum(b['events'] for b in manifest['blocks'])} unique events, 1,024 fresh roots; evaluation worker wall {summary['seconds']:.1f} s; CPU {summary['cpu_seconds']:.1f} s; peak RSS {summary['max_rss_kib']/1024**2:.2f} GiB")
        baseline=[r for r in a['no_edit_baselines'] if r['domain']==d]
        lines.append(f"  No-edit mean history F1 {100*np.mean([r['history']['f1'] for r in baseline]):.3f}%; recovery 0, damage 0.")
    lines += ['Worker timings exclude raw-source preparation and independent post-run auditing.',
              'No new training compute, annotations, natural-error adjudication or human-time measurement.',
              'All 768 saved repairs independently replayed; metrics recomputed; original events and exact undo verified.']
    (HERE/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'hypotheses.json',hypotheses)
    write(HERE/'claims.json',{'analysis_sha256':sha256(HERE/'analysis.json'),'claims':claims})
    (HERE/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.0),constrained_layout=True)
    for ax,d,title in zip(axes,DOMAINS,['Loans','Purchasing']):
        for arm,color,label in zip(ARMS,['#216b9b','#bc5a1b'],['Real8','Real8 + generated']):
            points=[get(d,64,arm,.99,p) for p in ['never_merge','guarded','unguarded']]
            for r,marker,pname in zip(points,['s','o','x'],['Never merge','Guarded','Unguarded']):
                ax.scatter(r['damage_rate']['mean']*1000,r['recovery_rate']['mean']*100,marker=marker,c=color,s=45,label=f'{label}: {pname}')
        ax.set_title(title); ax.set_xlabel('New damage per 1,000 initially correct events'); ax.grid(alpha=.2)
        ax.set_xlim(left=-.15)
    axes[0].set_ylabel('Initially incorrect events\nfully recovered (%)')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    fig.savefig(HERE/'figures/safety.pdf',bbox_inches='tight'); fig.savefig(HERE/'figures/safety.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    outputs=['paper_table_contrasts.tex','contrast_claims.json','paper_table_primary.tex','paper_table_safety.tex','RESULTS.txt','hypotheses.json','claims.json','figures/safety.pdf','figures/safety.png']
    write(HERE/'presentation.json',{'analysis_sha256':sha256(HERE/'analysis.json'),'script_sha256':sha256(__file__),
                                  'outputs':{n:sha256(HERE/n) for n in outputs}})
    print('\n'.join(lines))


if __name__=='__main__': main()
