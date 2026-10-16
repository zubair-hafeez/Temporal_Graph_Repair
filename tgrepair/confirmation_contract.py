"""Verify frozen fixed models or the predeclared target-local adaptation path."""
import json

from .data import sha256


def adaptation_names(task):
    base=f"confirmation_{task['domain']}_{task['observation_view']}_{task['variant']}_{task['fault_seed']}"
    return {'cache':base+'_pseudo','training':base+'_learner',
            'model':base+'_learner_self_supervised','calibration':base+'_generated_calibration'}


def verify_model(root,protocol,task,model_name,model_path,calibration_name):
    model_hash=sha256(model_path)
    score_path=root/'data/edit_learning'/f'{calibration_name}_validation_scores.npz'
    fixed=protocol['models'].get(model_name)
    if fixed is not None:
        if model_hash!=fixed['sha256']:raise ValueError('Frozen model changed')
        if fixed['calibrations'].get(calibration_name)!=sha256(score_path):
            raise ValueError('Calibration absent from or inconsistent with frozen model specification')
        return
    contract=protocol['target_local_adaptation'];names=adaptation_names(task)
    if task['variant']!='mixed' or model_name!=names['model'] or calibration_name!=names['calibration']:
        raise ValueError('Unregistered model or adaptation task')
    cache=root/'data/edit_learning'/names['cache']
    prep=json.loads((cache/'manifest.json').read_text())
    fitted=json.loads((root/'reports'/f"{names['training']}.json").read_text())
    scored=json.loads((root/'reports'/f"{names['calibration']}.json").read_text())
    if prep['observation_hash']!=task['observation_sha256'] or set(prep['target_types'])!=set(task['target_types']):
        raise ValueError('Adaptation was not prepared from this exact observation')
    if fitted['source_cache_manifest_hash']!=sha256(cache/'manifest.json') or fitted['config']['regime']!='self_supervised':
        raise ValueError('Adaptation training provenance mismatch')
    if len(fitted['models'])!=1 or fitted['models'][0]['model_hash']!=model_hash:
        raise ValueError('Adaptation model hash mismatch')
    validation=contract['validation_by_fault_seed'][str(task['fault_seed'])]
    if scored['model_hash']!=model_hash or scored['score_cache_hash']!=sha256(score_path):
        raise ValueError('Adaptation calibration hash mismatch')
    if scored['validation_manifest_hash']!=validation['manifest_hash'] or scored['config']['validation']!=validation['name']:
        raise ValueError('Adaptation used a different calibration collection')
    for report in [prep,fitted,scored]:
        for path,digest in contract['script_hashes'].items():
            if report['source_snapshot']['files'].get('research/'+path)!=digest:
                raise ValueError('Adaptation implementation changed: '+path)
