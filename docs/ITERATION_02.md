# Iteration 2 — stronger evidence, interacting faults, and generator integration

All outcomes here are development results. BPI 2017 and BPI 2019 remain unopened for model selection.

## Strengthening the participant baseline

The first structural comparison excluded all descriptive attributes. A richer, explicitly declared track now retains the source BPI 2012 requested amount as a repeated event-record field. It contains no case ID; many cases share the same amount. The full candidate pool and every matcher receive the same evidence. This is a denormalized-record observation condition, not a claim that the raw XES stores amount as an event-level field.

With 512 candidates and 1,000 query groups per run, a source-trained feature matcher recovers **97.3%** of independent gaps and **82.5%** of contiguous gaps in the tune partition. Candidate recall is 99.9% and 100%, respectively. This substantially weakens a paper based only on missing-link prediction. See `reports/matching_attributes_v2.json`.

A second probe samples 2,000 events without revealing corruption locations. Its observed graph combines 10% contiguous deletion, 10% random same-type wrong assignments among remaining links, and namespace fragmentation on 25% of eligible entities. The feature matcher sees current assignments explicitly but excludes the query event from its supporting history. It trains on fit cases and tunes on separate cases.

The selected development setting reaches 97.0% participant accuracy versus 81.2% for no change, preserves 1,623 of 1,624 originally correct sampled relations, and repairs 317 of 376 damaged relations. Equivalent aliases are accepted for this participant metric; it does not measure identity recovery. The initial run completed training but failed to serialize a NumPy integer in the report. Evaluation was recovered from the saved model and unchanged cache, without refitting. Both the partial report and `reports/deployment_matching_v1_recovered.json` are retained.

## Connected histories reveal a remaining problem

The whole development graph contains 38,777 events and 1,929 core applications. On this graph, the same participation matcher raises macro object-history retrieval F1 from **0.7445 to 0.8922**. It preserves 31,350 of 31,362 originally correct relations. This is a strong sequential baseline, not our proposed contribution. See `reports/sequential_development_v1.json`.

Simple temporal/context identity matching, with greedy or global bipartite assignment, reduces history quality when appended to that baseline. The best tested merging setting reaches only 0.8856. Identity fragments often cross resource handoffs, while wrong participant links contaminate apparent lifetimes. This motivates testing coupled restoration, but a weak identity heuristic alone cannot validate the gap. A learned identity matcher and a stronger sequential learned pipeline are still required.

An adaptation of the published OCIP attribute-variant detector also overmerges when only requested amount is available. This is an adaptation on a different, less descriptive source schema: the original notebook uses six attributes on BPI 2017 and reports clone detection, not our end-to-end repair metric. Do not present these numbers as a reproduction of its reported performance.

## Bounded factory implementation

The paper now uses `RF_Dataset_Factory/src/temporal_factory/` through a registered repair task plugin. A world execution and its observation policy are separate. The workflow supports branching/joining, batching, handoffs, retries, cancellation/return, and shared queues. Clean and damaged variants preserve the same true behavior. Prefix-only prediction provides a second task from the same execution.

Nine factory checks and fifteen research checks pass. A 32-world pilot v2 produced 3,638 events, 64 repair variants, and prediction tasks. Its schema/projection schedule was found to couple flat observations to one vocabulary profile; subsequent learning generation corrects that design. The pilot is retained. No generator calibration or synthetic-to-real benefit has been demonstrated. Source archives are now captured for subsequent runs so later edits do not erase the implementation used by an experiment.

## Decision

Continue to a bounded synthetic-training experiment. The proposed positive result must exceed the strong participant matcher at connected-history restoration and survive a learned identity/sequential baseline. Include flat and multi-object observations, varying mechanisms, legitimate attributes, unchanged controls, and actual exceptions. Keep the principal real targets sealed until the method and evidence policy are frozen. A graph model or generator that adds no value should be revised or removed from the claim.
