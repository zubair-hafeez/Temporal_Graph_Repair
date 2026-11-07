"""Write result-grounded manuscript findings with numerical provenance receipts."""
import json
from .run import HERE, ROOT, write
from tgrepair.data import sha256


def main():
    a=json.loads((HERE/'analysis.json').read_text()); receipts=[]
    def receipt(source,path,scale=1,digits=2):
        obj=json.loads((HERE/source).read_text())
        for k in path: obj=obj[k]
        value=obj*scale; literal=f'{value:.{digits}f}'
        receipts.append({'source':source,'sha256':sha256(HERE/source),'path':path,'scale':scale,'value':value,'literal':literal})
        return literal
    def mean(d,arm,k,policy='guarded',digits=2,scale=1):
        i=next(i for i,r in enumerate(a['aggregates']) if (r['domain'],r['budget'],r['precision'],r['arm'],r['policy'])==(d,64,.99,arm,policy))
        return receipt('analysis.json',['aggregates',i,k,'mean'],scale,digits)
    c=next(i for i,r in enumerate(a['contrasts']) if r['domain']=='bpi2017' and r['budget']==64 and r['precision']==.99 and r['left']==['real8_generated','guarded'] and r['right']==['real8','guarded'])
    def delta(k,part='mean',digits=2,scale=1):
        path=['contrasts',c,'metrics',k]+(['mean'] if part=='mean' else ['ci95',part])
        return receipt('analysis.json',path,scale,digits)
    rr=mean('bpi2017','real8','recovery',digits=1); rg=mean('bpi2017','real8_generated','recovery',digits=1)
    hr=mean('bpi2017','real8','damage'); hg=mean('bpi2017','real8_generated','damage')
    dr=delta('recovery',digits=1); rl=delta('recovery',0,digits=1); ru=delta('recovery',1,digits=1)
    dh=delta('damage'); hl=delta('damage',0); hu=delta('damage',1)
    df=delta('f1',scale=100); fl=delta('f1',0,scale=100); fu=delta('f1',1,scale=100)
    loss=delta('f1',scale=-100)
    oldharm=mean('bpi2017','real8','damage','unguarded')
    neverharm=mean('bpi2017','real8','damage','never_merge')
    nreal=mean('bpi2017','real8','recovery','never_merge',digits=1)
    oldr=mean('bpi2017','real8','recovery','unguarded',digits=2)
    rprec=mean('bpi2017','real8','recovery',digits=2)
    ltp=mean('bpi2017','real8','correct_pairs')
    sd=[receipt('seed_sensitivity.json',['seeds',i,'difference','recovery'],digits=1) for i in range(3)]
    pget=lambda arm,k,p='guarded':mean('bpi2019',arm,k,p)
    text=rf'''The prespecified loan result replicates (Table~\ref{{tab:fresh-primary}}). At64 verified bundles and$q=.99$, augmentation raises mean full-event recovery from{rr} to{rg} per block and reduces new damage from{hr} to{hg}. After averaging the three fixed model/fault runs within each block, recovery improves in all eight blocks. Fitted-seed sensitivity remains: the three seed-specific recovery differences, averaged across blocks, are{sd[0]}, {sd[1]} and{sd[2]} events. Thus one fitted-seed mean loses recovery; block uncertainty does not cover new training draws. The paired recovery difference is{dr} events[95\% interval{rl},{ru}], and the damage difference is{dh}[{hl},{hu}]. H1 meets both its directional and interval criteria. However, history F1 decreases by{loss} percentage points: its paired difference is{df}[{fl},{fu}]. The confirmed benefit concerns recovery and harm at this operating point, not uniform improvement in every endpoint.

Table~\ref{{tab:fresh-primary}} also retains the failures. At256 bundles, augmentation lowers mean history F1 and recovery and increases mean damage in both sources at both targets; not every damage interval excludes zero. On purchasing at64 bundles it reduces recovery and increases damage at both targets. Its small purchasing F1 gain at16 bundles and$q=.99$ accompanies fewer full-event recoveries and more damage. Thus generated data can add value beyond repeated corruption, but the evidence supports a narrow operating condition rather than a general verification-budget saving or a universal rule for when to stop generating data.

The never-merge control changes the safety interpretation (Tables~\ref{{tab:fresh-safety}} and~\ref{{tab:fresh-contrasts}}). For real-only loans, guarding reduces damage from{oldharm} to{hr}, while recovery changes from{oldr} to{rprec}. Never merging further reduces damage to{neverharm} with the same{nreal} recoveries, but gives up {ltp} true merged pairs and some history F1. On purchasing, guarded real-only/augmented policies retain{pget('real8','correct_pairs')}/{pget('real8_generated','correct_pairs')} true pairs per block. Their history F1 exceeds never merge, but their damage is{pget('real8','damage')}/{pget('real8_generated','damage')}, versus{pget('real8','damage','never_merge')}/{pget('real8_generated','damage','never_merge')}. The corresponding extra-harm intervals are positive. None of the four H2 comparisons establishes a beneficial merge policy without added harm over never merge.

The strict augmented loan model makes no merges under any of the three policies; all three produce identical outcomes. Its zero observed harm therefore does not demonstrate an incremental guard benefit. Moreover, real-only never merge still recovers{nreal} events versus{rg} for the augmented model, so the recovery advantage is not explained solely by false merges in the real-only comparator. Never merging eliminates false identity merges by construction, but participation repairs can still damage records. The choice between retaining fragmented histories and accepting residual merge risk remains an application decision; the measured frontier supplies evidence, not a deployment safety guarantee.

All768 saved repair programs passed independent replay, exact undo, event-content preservation and recomputed history/identity and cluster-expanded damage checks. The source-selection audit verified all older partition hashes and complete coverage of the declared new roots. No new fitting or threshold search was performed. Complete per-block/per-seed outcomes, conditional intervals and measured evaluation costs accompany the manuscript.
'''
    # Explicit prose spacing; never transform TeX paths, labels or metric names.
    import re
    text=re.sub(r'\b(At|at|from|to|by|is|versus|same|retains|retain|recovers|All)(?=[0-9-])',r'\1 ',text)
    text=text.replace('and$q','and $q').replace('events[95','events [95').replace('interval'+rl,'interval '+rl)
    # Readable separation around the remaining formatted measurements.
    text=re.sub(r'(damage|difference|recovery|are|and|of)(?=-?\d)',r'\1 ',text)
    text=text.replace(f'{dr} events[',f'{dr} events [').replace(f'{dh}[',f'{dh} [').replace(f'{df}[',f'{df} [')
    text=text.replace(f'{rl},{ru}',f'{rl}, {ru}').replace(f'{hl},{hu}',f'{hl}, {hu}').replace(f'{fl},{fu}',f'{fl}, {fu}')
    (ROOT/'revision/paper/confirmation_interpretation.tex').write_text(text)
    abstract=rf'''An operational history can lose evidence through missing links, wrong joins or fragmented identities. Generated workflows provide exact repair references, but their value must be tested against repeated corruption of the same verified real histories and against the errors introduced by repair. We compare eight real-history corruption views with the same data plus executable generated workflows at matched simulated verification budgets, including fitting and calibration. An exploratory study supplies input ablations and fixed models; a prospective confirmation evaluates36 unchanged fits in768 conditions on1,024 untouched root histories per source, using public loan and purchasing logs with controlled injected faults. At the prespecified64-bundle loan setting with a .99 calibration precision target, generated training raises mean fully recovered events from{rr} to{rg} per128-root block and reduces newly damaged events from{hr} to{hg}. The paired recovery gain is{dr}[{rl}, {ru}] under conditional95\% block-bootstrap intervals, although one of three fitted-seed means loses recovery and mean history F1 falls by{loss} percentage points. At256 bundles, augmentation has worse mean history F1, recovery and damage in both sources at both tested targets. Merge guards substantially reduce harm relative to unguarded decoding, but a never-merge baseline exposes their residual risk: no tested source/training combination gains from merging without added harm. These results establish conditional value for generated supervision and quantify the cost of identity recovery. They do not establish natural-error performance, human verification savings or automatic-repair safety.'''
    abstract=re.sub(r'\b(evaluates|in|on|prespecified|from|to|per|is|by|At|at|conditional)(?=[0-9-])',r'\1 ',abstract)
    abstract=abstract.replace(f'{dr}[',f'{dr} [')
    p=ROOT/'revision/paper/main.tex'; main=p.read_text(); start=main.index('\\begin{abstract}')+len('\\begin{abstract}');end=main.index('\\end{abstract}',start)
    main=main[:start]+'\n'+abstract+'\n'+main[end:]
    start=main.index('\\section{Conclusion}');end=main.index('\\input{references.tex}',start)
    conclusion=r'''\section{Conclusion}
Untouched-history confirmation supports one prespecified benefit of generated supervision beyond repeated corruption of verified histories: greater loan event recovery with less observed harm at a strict, limited-budget operating point. At 256 bundles, the fixed augmentation has worse mean history F1, recovery and damage in both sources at both targets. It also loses on the main purchasing comparisons. Conservative identity checks substantially reduce damage relative to unguarded merging, but never merge exposes a remaining tradeoff between complete histories and new errors. A strong repair evaluation therefore needs a competitive corruption baseline, a merge-abstention control, and separate recovery and harm endpoints. Natural-error validation, new institutions and measured verification effort remain necessary to establish field value.

'''
    main=main[:start]+conclusion+main[end:]
    main=main.replace('Its purchasing budget advantage is relative to a single-corruption real-only control; the repeated-corruption follow-up does not establish consistent added value at comparable harm. Neither study is a general sample-complexity result, a measured reduction in human work, or evidence of transfer to an unseen institution.', 'Its purchasing budget advantage is relative to a single-corruption real-only control. The fresh repeated-corruption confirmation supports the prespecified strict-loan recovery/harm benefit, while rejecting general augmentation superiority and a cost-free merge-safety benefit. These studies are not general sample-complexity results, measured reductions in human work, or evidence of transfer to an unseen institution.')
    main=main.replace('The follow-up uses a different bounded generated collection, selected-history context and one-pass decoding, so it cannot attribute differences from the earlier study solely to corruption count.', 'The repeated-corruption models use a different bounded generated collection, selected-history context and one-pass decoding from the earlier study; differences between studies cannot be attributed solely to corruption count.')
    p.write_text(main)
    # Cohort-size numbers already appear in the methods.
    for i in range(2):
        source='freshness.json'; value=json.loads((HERE/source).read_text())['domains'][i]['unique_events']
        receipts.append({'source':source,'sha256':sha256(HERE/source),'path':['domains',i,'unique_events'],
                         'scale':1,'value':value,'literal':f'{value:,}'})
    write(HERE/'narrative_claims.json',receipts)


if __name__=='__main__': main()
