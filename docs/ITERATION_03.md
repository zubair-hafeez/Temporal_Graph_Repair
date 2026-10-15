# Iteration 3: stronger identity learning and a failed transfer assumption

The source-trained identity matcher raises BPI 2012 tune history F1 from 0.8922 after participant repair to 0.9107. Its selected development operating point merges 134 pairs, of which 115 are correct, among 355 true fragmented pairs. Candidate recall is only 0.7634. The source learned baseline is substantially stronger than the initial heuristic and is retained.

The matching model trained only on the original generated worlds reaches 0.8959 after participant repair. On the raw mixed-error graph its best setting makes no edits. This does not support a synthetic transfer claim. Reports and trained models are preserved as negative development evidence.

The fit-only diagnosis finds median/99th-percentile events per entity of 11/84 in the real source, versus 5/7 in generated flat observations; activity and resource-partner diversity are also much lower. The next bounded experiment adds longer work routes, repeated work, start/completion lifecycle records, variable waiting times, population sizes, and observation-attribute projections. It is domain randomization motivated by development evidence, not calibrated reproduction of a real company. Main target tests remain sealed.

The first graph-edit model and matched MLP/boosted controls train on the original generated collection while expanded data are prepared. Their input construction uses only observed graphs; training-only references label candidates and produce partial repair states. The decoder must discover edit locations and may stop without complete restoration. An improvement in action classification alone is insufficient: whole-history recovery, clean-history preservation, identity merging and runtime will decide whether this method survives.

The initial CUDA attempt failed inside the sandbox before training. The existing training script was then authorized through the environment's scoped approval mechanism. The failure log remains in reports/edit_temporal_v1.log. No drivers, dependencies, or other processes were changed.
