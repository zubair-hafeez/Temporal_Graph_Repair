"""Measure executable-world/fault creation separately from cached feature construction."""
from collections import Counter, defaultdict
import json
import platform
import time

import numpy as np
import sklearn

from tgrepair.data import load_graph, sha256
from .method import corrupt, workflow
from .run import FIT_VARIANTS, HERE, verify, write


def main():
    verify()
    cpu_model = next((line.split(':', 1)[1].strip() for line in open('/proc/cpuinfo') if line.startswith('model name')), 'unknown')
    start = time.perf_counter(); cpu = time.process_time()
    worlds = []
    for i in range(32):
        clean, count = workflow(341000 + i)
        observed, reference = corrupt(clean, {'Root', 'Item'}, 342000 + i, FIT_VARIANTS[i % 8])
        worlds.append((clean, observed, reference, count))
    wall_seconds, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu
    activities = Counter(); events = roots = 0; overlap = []; ties = []
    for i, (clean, observed, reference, count) in enumerate(worlds):
        folder = HERE / 'data/generated' / f'world{i}'
        assert clean == load_graph(folder / 'clean.json.gz')
        assert observed == load_graph(folder / 'observed.json.gz')
        assert json.loads(json.dumps(reference)) == json.loads((folder / 'reference.json').read_text())
        activities.update(count); events += len(clean.events)
        histories = defaultdict(list)
        for e in clean.events:
            for o in e.participants:
                if clean.objects[o] == 'Root': histories[o].append(e.time)
        roots += len(histories)
        boundaries = sorted((t, sign) for ts in histories.values() for t, sign in [(min(ts), 1), (max(ts), -1)])
        active = peak = 0
        for _, sign in boundaries:
            active += sign; peak = max(peak, active)
        overlap.append(peak)
        times = Counter(e.time for e in clean.events)
        ties.append(sum(n for n in times.values() if n > 1) / len(clean.events))
    caches = {mode: json.loads((HERE / 'data' / f'generated_{mode}.json').read_text())
              for mode in ['full', 'no_time', 'no_context', 'no_attributes', 'no_history']}
    write(HERE / 'generation_benchmark.json', {
        'scope': 'Fresh execution and fault creation without disk I/O; feature timings are the original cache runs.',
        'worlds': 32, 'roots': roots, 'events': events, 'activities': dict(activities),
        'peak_overlapping_roots_per_world': overlap, 'fraction_events_sharing_a_timestamp_per_world': ties,
        'generation_wall_seconds': wall_seconds, 'generation_cpu_seconds': cpu_seconds,
        'feature_cache_times': {mode: {k: value[k] for k in ['seconds', 'cpu_seconds', 'actions']} for mode, value in caches.items()},
        'cpu_model': cpu_model, 'python': platform.python_version(), 'numpy': np.__version__,
        'sklearn': sklearn.__version__, 'all_worlds_and_references_reproduced': True,
        'generated_manifest_sha256': sha256(HERE / 'generated_manifest.json'), 'script_sha256': sha256(__file__)})


if __name__ == '__main__':
    main()
