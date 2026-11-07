"""Count the real reference histories used by the fixed confirmation models."""
import json
from .run import HERE, OLD, DOMAINS, write
from tgrepair.data import sha256


def main():
    text=[r'\begin{table}[t]',r'\centering\small',
        r'\caption{Real reference access for the repeated-corruption models reused in confirmation. Ranges cover three fixed allocation seeds; both training arms receive the same histories. Event and target-entity counts cover fitting plus calibration. Artificial corruptions add no new real histories. These counts measure simulated reference access, not human time.}',
        r'\label{tab:annotation-cost}',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Source & Total bundles & Fit/calibration & Events & Target entities \\ \midrule']
    receipts=[]
    for d,label in zip(DOMAINS,['Loans','Purchasing']):
        path=OLD/'results'/d/'summary.json'; a=json.loads(path.read_text())
        for b in [16,64,256]:
            rows=[r for r in a['allocations'] if r['budget']==b]; assert len(rows)==3
            cells=[label,str(b),f'{b*3//4}/{b//4}']
            for field in ['events','target_entities']:
                values=[sum(r[role][field] for role in ['fit','calibration']) for r in rows]
                lo,hi=min(values),max(values); literal=f'{lo:,}--{hi:,}'
                cells.append(literal)
                receipts.append({'source':str(path.relative_to(OLD)),'source_sha256':sha256(path),
                                 'budget':b,'field':field,'values':values,'literal':literal})
            text.append(' & '.join(cells)+r' \\')
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (HERE/'paper_table_budget.tex').write_text('\n'.join(text)+'\n')
    write(HERE/'budget_claims.json',receipts)


if __name__=='__main__': main()
