# Iteration 6: transfer restored through varied observation scenarios

All outcomes below are development results. Reserved real-log and independent-simulator targets remain unopened.

## Completed paired comparison

| Training / observation condition | BatteryVault | ClaimStream | BPI2012 tune, denormalized amount |
|---|---:|---:|---:|
| Damaged input | 0.768167 | 0.761670 | 0.744456 |
| Original generated v2, native fields | 0.737127 | 0.751895 | 0.837963 |
| Original model, attributes hidden manually | 0.854300 | 0.884421 | Not run |
| Original model, new generated calibration only | 0.796371 | 0.797792 | Not run |
| Varied-attribute generated training v3 | **0.880412** | **0.919719** | 0.807426 |
| Original model, automatic stable-field view | 0.858824 | 0.892031 | **0.839250** |

The varied-attribute learner uses the same workflows, events, faults, model family and query budgets as its original-training control. It benefits both independent simulation domains even compared with changing calibration alone. It loses 0.030537 history F1 relative to the old model on the explored BPI amount track. No uniformly dominant model or universal improvement is established. The automatic field filter is a simpler competing observation policy; its development-selected stability threshold is 0.5.

The new learner makes zero edits to both clean independent counterparts: all 22,085 BatteryVault and 24,929 ClaimStream event assignments and histories remain intact. On the mixed-fault inputs it preserves 17,624/17,708 and 18,267/18,325 initially correct events. These clean tests support the measured operating point only, not a universal safety guarantee.

Identity recall improves to 69/246 (0.280488) in BatteryVault and 199/403 (0.493797) in ClaimStream. There are 9 and 4 incorrect predicted identity pairs respectively. Genuine exception-evidence F1 is 0.924686 and 0.964828, compared with damaged inputs 0.869986 and 0.837409. Correct boundary-time retrieval also improves. Reports retain per-type scores and incomplete histories.

## Stronger unlabelled-source controls

With the original generated calibration held common, extra-corruption self-supervision on imperfect BPI fit observations reaches 0.755124 history F1; adding generated training reaches 0.832864. Pure generated v2 reaches 0.837963. This isolates a training-data contribution beyond the earlier pseudo-calibration abstention result. It does not establish target-local superiority on the other domains.

New target-local controls therefore learn from each imperfect external observation itself, without opening its reference file. The input is split by time, extra faults provide noisy pseudo labels, and the same generated validation selects thresholds. This gives the comparator transductive access to unlabelled target records. Existing errors remain in its pseudo targets; those errors are not secretly corrected using evaluation labels.

## Repetition and remaining checks

Two independent 512-world training / 96-world validation repetitions are running. Each repetition has paired original and varied-attribute observations and separately fitted boosted models. Actual simulated worlds and faults vary across repetitions; a mere classifier random-state change is not counted as a new experiment.

Refitted feature controls remove temporal scoring features or context-overlap scoring features while holding candidate pools fixed. Temporal/contextual retrieval remains available, so these controls must not be described as entirely time-free or graph-free systems. Independent pipeline-fault stress runs are also pending; that fault family has not entered training.

An observation-only scale diagnostic finds that BPI fit resource contexts are shared by many more target handles than the small training worlds (median 519 versus a median-world 14.25). This is a plausible coverage limitation, not a demonstrated cause of low identity recall. No larger-population generation change was made. Expanding the generator is deferred while the simpler useful candidate undergoes confirmation checks.

Reserved real-source adapters have fixture tests for prefix exclusions, explicit offer links, identifier-field removal and bounded-memory top-level JSON streaming. They have not yet read reserved records. The complete general factory remains outside this paper.

## Completed stress and feature controls

The target-local self-supervised controls, calibrated on the same generated validation, reach 0.771567 on BatteryVault and 0.776398 on ClaimStream. The generated-only learner reaches 0.880412 and 0.919719 respectively. These baselines see their own imperfect target observations but no clean target reference labels.

The independent pipeline faults remain substantially harder. The generated learner raises BatteryVault F1 from 0.775792 to 0.807484, and ClaimStream from 0.763679 to 0.798482. Identity precision is 38/45 and 42/68; the latter has 26 incorrect predicted pairs. Preserve this failure boundary. Do not describe the model as reliably merging arbitrary cross-system namespaces.

On the same 24 generated validation worlds, removing temporal scoring features reduces F1 from 0.825555 to 0.781345; removing context-overlap scoring reduces it to 0.768780. On BPI tune, the corresponding outcomes are 0.745666 and 0.810402 versus 0.807426 for the complete scorer. Timing helps markedly in the explored real loan setting; context scoring does not improve that setting. Candidate retrieval retains both kinds of evidence in these ablations.

Both independent training-replication pipelines are complete. Matched GraphSAGE, MLP and temporal-message controls were also retrained for 16 epochs on the revised attribute data; their application evaluations remain in progress. No abandoned neural result is relabeled as an exact published-method reproduction.

## Confirmation transition

`CONFIRMATION_PROTOCOL_V1.json` was frozen before reserved evaluation with SHA256 `a3e12b092cc7670f8b4d2c7190818eb2c04ea88e7ae4a2afb20bdffac691c25d`. It pins model/calibration artifacts, source files, adapters and decoder implementation; predeclares three paired training/fault trials and the secondary controls; and constrains target-local adaptation to the exact imperfect observation and frozen scripts. Reserved BPI2012-check/BPI2017/BPI2019 preparation and the 96-world held-out composition collection were then started. Their outcomes cannot now be used to retune this frozen method while retaining a confirmation claim.

The four reserved independent-simulator downloads timed out through both the API and direct-file URLs, including approved network retries. No records from those four sources have been decoded. Local real-source confirmation proceeds while downloads remain pending. This is an acquisition issue, not evidence about method performance.

The research suite passes 41 tests and the bounded temporal-factory suite passes 10 tests. The paper PDF compiles locally; it remains an internal development manuscript until confirmation, application case analysis and the final evidence review are complete.
