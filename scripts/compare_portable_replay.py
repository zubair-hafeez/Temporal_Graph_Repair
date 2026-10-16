"""Compare a relocated full-task run with its original frozen evaluation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tgrepair.data import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--portable-root', required=True)
    parser.add_argument('--original', required=True)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    portable = Path(args.portable_root).resolve()
    source = ROOT/'reports'/f'{args.original}.json'
    replay = portable/'reports'/f'{args.replay}.json'
    original_report = json.loads(source.read_text())
    replay_report = json.loads(replay.read_text())
    for report in [original_report, replay_report]:
        if report['status'] != 'completed_confirmation_evaluation':
            raise ValueError('Both evaluations must have completed')
    fields = ['before', 'after', 'queries_before', 'queries_after', 'thresholds', 'calibration', 'calibration_cache_hash', 'edit_count', 'attribute_view']
    for field in fields:
        if original_report[field] != replay_report[field]:
            raise ValueError('Replay differs: '+field)
    for field in ['observation_sha256', 'reference_sha256', 'source_sha256', 'target_types']:
        if original_report['task'][field] != replay_report['task'][field]:
            raise ValueError('Replay task differs: '+field)
    a = ROOT/'data/decoding'/args.original
    b = portable/'data/decoding'/args.replay
    for name in ['repaired.json.gz', 'clusters.json']:
        if sha256(a/name) != sha256(b/name):
            raise ValueError('Replayed graph/identity artifact differs')
    first = json.loads((a/'audit.json').read_text())
    second = json.loads((b/'audit.json').read_text())
    if first['rounds'] != second['rounds']:
        raise ValueError('Full edit sequence differs')
    result = {'status': 'full_relocated_task_replay_passed', 'original': args.original, 'replay': args.replay,
              'original_report_hash': sha256(source), 'replay_report_hash': sha256(replay),
              'script_hash': sha256(__file__), 'checked_fields': fields,
              'exact_repaired_graph_and_clusters': True, 'exact_edit_program': True,
              'limit': 'Same Python environment and hardware; source/task paths relocated. This is replay, not independently retrained model evidence.'}
    output = ROOT/'reports'/f'{args.output}.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
