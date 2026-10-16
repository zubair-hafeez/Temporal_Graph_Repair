"""Keep post-freeze calibration sensitivity separate from primary confirmation."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256
from write_confirmation_tables import table, cell, NAMES


def summarize(values):
    return {'values': values, 'defined_trials': len(values), 'mean': statistics.mean(values),
            'min': min(values), 'max': max(values)}


def main():
    declaration_path = ROOT/'docs/CALIBRATION_SENSITIVITY_V1.json'
    declaration = json.loads(declaration_path.read_text())
    suite_path = ROOT/'reports/calibration_sensitivity_v1.json'
    suite = json.loads(suite_path.read_text())
    if suite['status'] != 'completed_post_freeze_calibration_sensitivity':
        raise ValueError('Sensitivity outcomes incomplete')
    if suite['declaration_hash'] != sha256(declaration_path):
        raise ValueError('Sensitivity declaration changed')
    rows, sources = [], {str(declaration_path.relative_to(ROOT)): sha256(declaration_path),
                         str(suite_path.relative_to(ROOT)): sha256(suite_path)}
    for job in declaration['jobs']:
        path = ROOT/'reports'/f"{job['output']}.json"
        report = json.loads(path.read_text())
        outcome = next(x for x in suite['outcomes'] if x['output'] == job['output'])
        if sha256(path) != outcome['report_hash']:
            raise ValueError('Sensitivity result changed')
        if report['config']['calibration'] != declaration['calibration'] or report['pins']['model_hash'] != declaration['model_hash']:
            raise ValueError('Wrong frozen model/calibration')
        original_name = f"confirm_{job['domain']}_{job['view']}_mixed_{job['seed']}_source256"
        original_path = ROOT/'reports'/f'{original_name}.json'
        original = json.loads(original_path.read_text())
        if original['status'] != 'completed_confirmation_evaluation' or original['pins']['task_hash'] != report['pins']['task_hash']:
            raise ValueError('Original comparison incomplete or mismatched')
        if original['before'] != report['before'] or original['pins']['model_hash'] != report['pins']['model_hash']:
            raise ValueError('Calibration comparison changes inputs/weights')
        def damage(value):
            return value['after']['events']['originally_correct_events']-value['after']['events']['preserved_correct_events']
        rows.append({'domain': job['domain'], 'seed': job['seed'], 'output': job['output'],
                     'before_f1': report['before']['f1'], 'source_calibration_f1': original['after']['history']['f1'],
                     'generated_calibration_f1': report['after']['history']['f1'],
                     'source_calibration_new_damage': damage(original), 'generated_calibration_new_damage': damage(report),
                     'source_thresholds': original['thresholds'], 'generated_thresholds': report['thresholds']})
        for p in [path, original_path]:
            sources[str(p.relative_to(ROOT))] = sha256(p)
    groups, table_rows = [], []
    fields = ['before_f1', 'source_calibration_f1', 'generated_calibration_f1',
              'source_calibration_new_damage', 'generated_calibration_new_damage']
    for domain in ['bpi2017', 'bpi2019', 'bpi2012_check']:
        selected = sorted([x for x in rows if x['domain'] == domain], key=lambda x: x['seed'])
        if [x['seed'] for x in selected] != [91001, 91002, 91003]:
            raise ValueError('Missing or duplicate sensitivity trial')
        metrics = {f: summarize([x[f] for x in selected]) for f in fields}
        groups.append({'domain': domain, 'metrics': metrics})
        table_rows.append([NAMES[domain], cell(metrics['before_f1']), cell(metrics['source_calibration_f1']),
                           cell(metrics['generated_calibration_f1']),
                           '/'.join(str(int(x)) for x in metrics['source_calibration_new_damage']['values']),
                           '/'.join(str(int(x)) for x in metrics['generated_calibration_new_damage']['values'])])
    result = {'status': 'complete_post_freeze_calibration_analysis', 'scope': declaration['scope'],
              'source_reports': sources, 'script_hash': sha256(__file__), 'trials': rows, 'groups': groups}
    output = ROOT/'reports/calibration_sensitivity_evidence_v1.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    content = table(['Source', 'Damaged', 'Source calibration', 'Generated calibration', 'Source: new damage', 'Generated: new damage'], table_rows,
        'Exploratory calibration sensitivity declared after inspecting primary transfer results. The source-trained weights are identical in both conditions; both calibration caches were frozen before confirmation, but this comparison was added later. F1 cells show mean and observed range across three fault seeds. Counts list newly damaged events in each trial. These results supplement rather than replace the originally declared source-calibrated control.',
        'tab:calibration-sensitivity')
    (ROOT/'paper/tables/calibration_sensitivity.tex').write_text(content)
    print(json.dumps(groups))


if __name__ == '__main__':
    main()
