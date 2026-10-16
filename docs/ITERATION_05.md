# Iteration 5: observation mismatch and stronger no-clean-label controls

## Independent failure, retained

The initial generated-only v2 policy degrades native-field BatteryVault history F1 from 0.768167 to 0.737127 and ClaimStream from 0.761670 to 0.751895. It removes many correct associations and makes almost no additions. These are independent-simulator development failures, not final confirmation results. They must remain in the research record and motivate method changes rather than being omitted from a favorable comparison.

A paired diagnostic samples 256 events per domain and holds native candidate pools fixed while hiding descriptive fields. ADD average precision rises from 0.2253 to 0.8210 for BatteryVault and from 0.4129 to 0.8273 for ClaimStream. Native candidate recall is 0.9926 and 1.0 respectively. This points to score/observation mismatch rather than inadequate candidate coverage in that sample. It does not prove that every native field is uninformative.

Whole-window structural-only runs with the unchanged model and generated thresholds reach 0.854300 and 0.884421. Clean counterparts remain at 0.999451 and 0.999826. The structural view retains activities, time and event-object context, including anonymized native actors. Identity merges are 15/15 correct for BatteryVault and 97/97 for ClaimStream, but recalls remain only 0.0610 and 0.2407. Preserve this incompleteness.

Corrected evaluator-only exception-evidence F1 improves from 0.869986 to 0.924382 and 0.837409 to 0.950953 under structural repair. The initial structural runs mistakenly obtained exception flags from the attribute-hidden learner view, making their exception-positive set empty. `external_application_queries_v2.json` repairs that evaluation label source using pinned native parents; original history and cancellation metrics are unaffected. Future driver runs separate query labels from the learner view.

## Paired generation change

The v5 attribute-scenario collections reuse the exact v4 executions, visible types, events, object handles, injected fault labels and execution groups. Only simulated descriptive fields change. Four profiles retain original observations, add activity-dependent context, add varying measurements, or mix incidental categories, time bands and measurements. Fields are not generated from target identifiers. The added scenarios reflect a development diagnosis; no final-target outcomes were inspected.

An integrity audit checks 873,154 train-variant event records and 201,900 validation-variant records, including exact reference equality and clean/damaged attribute alignment through private execution ancestry. There are 436,577/100,950 visible clean events. The generation manifest's legacy `world_events` field counts these visible events, not the larger full-execution counts used in v4. This naming difference does not imply different executions or a volume increase.

The learning caches keep the same 896 train / 120 validation state schedule, event-query budget and candidate budgets. Candidate identity pairs can change with observed attributes, so this is a matched-world/query-budget comparison, not an identical binary-example count claim. The new boosted model has generated-validation AP 0.6744 ADD, 0.8887 REMOVE and 0.5268 MERGE. On that same new validation cache, the unchanged old model has AP 0.2931, 0.7174 and 0.2582. Full application transfer remains in progress.

Both old and new models can use the same expanded calibration collection. The old-model/new-calibration run separates changing the operating threshold from changing training observations. Do not attribute all improvements to training without this control.

An observation-only field-stability diagnostic was subsequently integrated into a separately evaluated policy. Its 0.5 threshold retains BPI's RequestedAmount and drops changing native fields in the two independent development logs. Completed results are recorded in ITERATION_06.md. It is a development candidate, not a target-selected final policy.

## Annotation and no-clean-label controls

Source-only versus source-plus-generated history F1 is 0.835864/0.891191 at 16 verified training histories, 0.832147/0.890525 at 64, and 0.875330/0.889635 at 256. Additional source-development calibration labels are shared by both arms. This is not total annotation cost, and performance is not monotonic in budget.

The new self-supervised control receives only a damaged fit observation, adds independent extra faults, and learns to undo those extra faults. Existing input errors remain pseudo-reference errors. Train/validation use disjoint time windows but can share entities. No original clean source labels reach fitting or calibration. At a 95% pseudo-validation precision criterion, the pure control abstains entirely; generated augmentation enables removals and reaches 0.772054 from 0.744456, with no full-event recoveries. These outcomes alone would make a weak comparator because pseudo-label noise constrains calibration. Stronger controls therefore keep the same models and calibrate them on the same generated validation used by the synthetic-only model. Those runs are in progress and must be reported.

## Publication status

`paper/main.pdf` compiles from a cached local TeX installation and is explicitly marked an internal development draft. Figures are measured development diagnostics, not confirmation. The closest-prior audit now includes Event Knowledge Graph inference, FLAWD and algebraic repair languages; graph repair, fault injection and edit programs are not new by themselves. The paper still needs a surviving application result, repeated independent confirmation, compatible published-method comparisons, reproducibility packaging and a critical final review.
