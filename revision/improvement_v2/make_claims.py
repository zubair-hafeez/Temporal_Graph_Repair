"""Pin the principal new manuscript numbers to independently audited results."""
import json

from .run import HERE, write
from tgrepair.data import sha256


def main():
    data = json.loads((HERE / 'analysis.json').read_text())
    claims = []
    def add(domain, arm, field, literal, precision=.95, mode='full', variant='mixed', safe=True, budget=64):
        index, row = next((i, r) for i, r in enumerate(data['aggregates'])
                          if r['domain'] == domain and r['budget'] == budget and r['arm'] == arm
                          and r['precision'] == precision and r['mode'] == mode and r['variant'] == variant and r['safe'] == safe)
        claims.append({'source': 'analysis.json', 'source_sha256': sha256(HERE / 'analysis.json'),
                       'field_path': ['aggregates', index, field, 'mean'], 'value': row[field]['mean'], 'literal': literal})
    for arm, field, literal in [('real8', 'recovery', 'from 30.3 to 67.0'),
                                ('real8_generated', 'recovery', 'from 30.3 to 67.0'),
                                ('real8', 'damage', 'zero observed new damage in either arm'),
                                ('real8_generated', 'damage', 'zero observed new damage in either arm')]:
        add('bpi2017', arm, field, literal, precision=.99)
    add('bpi2017', 'real8', 'recovery', 'from 236.7 to 108.3')
    add('bpi2017', 'real8_generated', 'recovery', 'from 236.7 to 108.3')
    add('bpi2017', 'real8_generated', 'recovery', 'from 108.3 to 7.7')
    add('bpi2017', 'real8_generated', 'recovery', 'from 108.3 to 7.7', mode='no_time')
    add('bpi2017', 'real8', 'damage', 'from 39.7 to 0.0', precision=.99, safe=False)
    add('bpi2017', 'real8', 'damage', 'from 39.7 to 0.0', precision=.99)
    add('bpi2017', 'real8', 'correct_pairs', 'from 5.3 to 1.7', precision=.99, safe=False)
    add('bpi2017', 'real8', 'correct_pairs', 'from 5.3 to 1.7', precision=.99)
    for safe in [False, True]:
        add('bpi2019', 'real8_generated', 'damage', 'from 10.3 to 4.7', precision=.99, safe=safe)
        add('bpi2019', 'real8_generated', 'recovery', 'from 307.7 to 308.7', precision=.99, safe=safe)
    add('bpi2019', 'real8_generated', 'damage', 'damages 5.3 events on clean inputs', variant='clean')
    add('bpi2019', 'real8_generated', 'recovery', '54.0 events while damaging 39.3', variant='pipeline')
    add('bpi2019', 'real8_generated', 'damage', '54.0 events while damaging 39.3', variant='pipeline')
    for domain, literal in [('bpi2017', '3.5/5.7 seconds'), ('bpi2019', '4.6/7.1 for purchasing')]:
        for arm in ['real8', 'real8_generated']:
            add(domain, arm, 'train_seconds', literal)
    for key, literal in [('conditions', '156 saved repairs'), ('models_with_replayed_calibration', '60 calibration computations')]:
        claims.append({'source': 'analysis.json', 'source_sha256': sha256(HERE / 'analysis.json'),
                       'field_path': [key], 'value': data[key], 'literal': literal})
    generation = json.loads((HERE / 'generation_benchmark.json').read_text())
    for path, literal in [(['worlds'], '32 executable worlds'), (['roots'], '512 roots'), (['events'], '4,507 events'),
                          (['activities', 'cancel'], '262 cancellations'),
                          (['feature_cache_times', 'full', 'seconds'], 'cache costs 10.1 seconds')]:
        value = generation
        for key in path: value = value[key]
        claims.append({'source': 'generation_benchmark.json', 'source_sha256': sha256(HERE / 'generation_benchmark.json'),
                       'field_path': path, 'value': value, 'literal': literal})
    write(HERE / 'claims.json', claims)


if __name__ == '__main__':
    main()
