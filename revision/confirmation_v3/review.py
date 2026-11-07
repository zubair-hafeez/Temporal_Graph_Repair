"""Verify current manuscript, complete confirmation, claims and retained evidence."""
import json
import re
from tgrepair.data import sha256
from revision.confirmation import verify_protocol as verify_original_revision
from .run import HERE, ROOT, verify, verify_cohort, DOMAINS, write


def main():
    protocol=verify(); verify_original_revision()
    initial=json.loads((HERE/'protocol_initial.json').read_text())
    assert sha256(HERE/'run_initial.py')==initial['implementation']['revision/confirmation_v3/run.py']
    assert protocol['amendments'][0]['prior_protocol_sha256']==sha256(HERE/'protocol_initial.json')
    for key,value in initial.items():
        if key!='implementation': assert protocol[key]==value
    for key,value in initial['implementation'].items():
        if key!='revision/confirmation_v3/run.py': assert protocol['implementation'][key]==value
    for domain in DOMAINS: verify_cohort(domain)
    fresh=json.loads((HERE/'freshness.json').read_text())
    assert fresh['status']=='passed' and fresh['script_sha256']==sha256(HERE/'freshness.py')
    for d in fresh['domains']: assert d['cohort_sha256']==sha256(HERE/'data'/d['domain']/'cohort.json')
    sensitivity=json.loads((HERE/'seed_sensitivity.json').read_text())
    assert sensitivity['per_trial_sha256']==sha256(HERE/'per_trial.csv')
    assert sensitivity['script_sha256']==sha256(HERE/'seed_sensitivity.py')
    assert sensitivity['positive_recovery_seed_means']==2 and sensitivity['negative_recovery_seed_means']==1
    a=json.loads((HERE/'analysis.json').read_text())
    assert a['conditions']==768 and a['status']=='all_768_conditions_independently_audited'
    assert a['analysis_code_sha256']==sha256(HERE/'analyze.py')
    for name,digest in a['checked_files'].items(): assert sha256(HERE/name)==digest,name
    old=ROOT/'revision/improvement_v2'
    oldanalysis=json.loads((old/'analysis.json').read_text())
    for name,digest in oldanalysis['checked_files'].items(): assert sha256(old/name)==digest,name
    b=json.loads((HERE/'build.json').read_text())
    assert b['pages']<=15 and not b['overfull']
    assert b['pdf_sha256']==sha256(ROOT/'revision/paper/main.pdf')
    assert b['analysis_sha256']==sha256(HERE/'analysis.json')
    for name,digest in b['inputs'].items(): assert sha256(ROOT/name)==digest,name
    visual=json.loads((HERE/'visual_review.json').read_text())
    assert visual['pdf_sha256']==b['pdf_sha256']
    assert visual['inspected_pages']==list(range(1,b['pages']+1))
    assert visual['no_clipped_or_overlapping_content']
    presentation=json.loads((HERE/'presentation.json').read_text())
    assert presentation['analysis_sha256']==sha256(HERE/'analysis.json')
    for name,digest in presentation['outputs'].items(): assert sha256(HERE/name)==digest,name
    text='\n'.join(p.read_text() for p in (ROOT/'revision/paper').glob('*.tex'))
    text+='\n'+'\n'.join(p.read_text() for p in HERE.glob('paper_table_*.tex'))
    keys=set(re.findall(r'\\bibitem\{([^}]+)\}',text))
    cited={k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')}
    assert cited<=keys
    assert '\\pending{' not in text
    claims=json.loads((HERE/'claims.json').read_text())
    assert claims['analysis_sha256']==sha256(HERE/'analysis.json')
    for c in claims['claims']:
        v=a['aggregates'][c['aggregate_index']][c['field']]['mean']*c['scale']
        assert abs(v-c['value'])<1e-12 and f"{v:.{c['digits']}f}"==c['literal']
        assert c['literal'] in text
    contrastclaims=json.loads((HERE/'contrast_claims.json').read_text())
    assert contrastclaims['analysis_sha256']==sha256(HERE/'analysis.json')
    for c in contrastclaims['claims']:
        m=a['contrasts'][c['contrast_index']]['metrics'][c['field']]
        assert [m['mean'],*m['ci95']]==c['values']
        assert [f"{v*c['scale']:.{c['digits']}f}" for v in c['values']]==c['literals']
        assert all(v in text for v in c['literals'])
    budgetclaims=json.loads((HERE/'budget_claims.json').read_text())
    for c in budgetclaims:
        source=old/c['source']; assert sha256(source)==c['source_sha256']
        rows=[r for r in json.loads(source.read_text())['allocations'] if r['budget']==c['budget']]
        values=[sum(r[role][c['field']] for role in ['fit','calibration']) for r in rows]
        assert values==c['values'] and c['literal'] in text
    narrative=json.loads((HERE/'narrative_claims.json').read_text())
    for c in narrative:
        source=HERE/c['source']; assert sha256(source)==c['sha256']
        v=json.loads(source.read_text())
        for k in c['path']: v=v[k]
        assert abs(v*c.get('scale',1)-c['value'])<1e-12
        assert c['literal'] in text
    tests=(HERE/'tests.log').read_text(); assert '66 passed' in tests
    write(HERE/'review.json',{'status':'confirmation_paper_ready_for_author_review','pages':b['pages'],
        'confirmation_conditions':768,'fixed_models':36,'new_roots_per_source':1024,
        'contrast_claim_receipts':len(contrastclaims['claims']),'budget_claim_receipts':len(budgetclaims),'table_claim_receipts':len(claims['claims']),'narrative_claim_receipts':len(narrative),
        'pdf_sha256':b['pdf_sha256'],'protocol_sha256':sha256(HERE/'protocol.json'),
        'analysis_sha256':sha256(HERE/'analysis.json'),'build_sha256':sha256(HERE/'build.json'),
        'visual_review_sha256':sha256(HERE/'visual_review.json'),'tests_sha256':sha256(HERE/'tests.log'),
        'old_scientific_implementations_and_results_unchanged':True,
        'freshness_sha256':sha256(HERE/'freshness.json'),
        'scope':'Untouched target histories within known source processes; controlled faults; no independent natural errors, human annotation time, or external submission.',
        'delivery_files':{n:sha256(HERE/n) for n in ['RESULTS.txt','REPRODUCE.txt','per_trial.csv','hypotheses.json','claims.json','narrative_claims.json','seed_sensitivity.json']}})
    print(json.dumps({'status':'confirmation_paper_ready_for_author_review','pages':b['pages'],'conditions':768}))


if __name__=='__main__': main()
