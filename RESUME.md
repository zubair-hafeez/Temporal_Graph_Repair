# Resume temporal graph repair research

The user authorized autonomous end-to-end paper development through final review. The goal is active/incomplete; no external submission. Preserve the previous paper, negative results and concurrent factory work. RF is excluded. No subagents are authorized.

## Current transition

Read docs/ITERATION_06.md and docs/CONFIRMATION_PROTOCOL_V1.json. The latter is frozen before reserved evaluation, SHA256 a3e12b092cc7670f8b4d2c7190818eb2c04ea88e7ae4a2afb20bdffac691c25d. Do not change frozen model/calibration/adapter/decoder files or retune from final results while claiming confirmation. Genuine bug fixes need an explicit preserved amendment and disclosure.

The principal learner is edit_feature_attributes_v3, with independently generated repeats edit_feature_attributes_repeat1/2. Paired original-observation controls are edit_feature_v2 (new attribute calibration) and edit_feature_plain_repeat1/2. All three actual world-generation pipelines are complete. Scope: executable training data for offline event-object history repair, not new neural architecture, a generic universal factory, or naturally occurring error recovery.

Main development: battery F1 .768167→.880412; claims .761670→.919719. Changing calibration only gives .796371/.797792. Both native clean counterparts receive zero edits. BPI2012 tune drops from old .837963 to .807426; automatic stable-field view .839250/.858824/.892031 on BPI/battery/claims. Keep the trade-off. Target-local extra-corruption controls reach .771567/.776398. Pipeline stress reaches .807484/.798482, with identity precision 38/45 and 42/68: false merges are a material limitation.

Timing/context scoring ablations are explicit, refitted, and retain temporal/context retrieval. Generated 24-world F1 .825555 full, .781345 without timing scoring, .768780 without context scoring; BPI .807426/.745666/.810402. No unconditional context-feature benefit. Matched revised GraphSAGE/MLP/temporal models completed 16 epochs; selected epoch 10 for all. Generated application runs and two-repeat external transfer runs remain live in STATE.json.

## Reserved tasks and running confirmation

Reserved local sources have now been prepared under the frozen protocol. BPI2017: 1,024 hash-selected applications, 1,398 explicit offers, 38,822 events, excluding first 500 audited applications/offers. BPI2019: 1,024 hash-selected POs, 3,301 items, 20,396 events, excluding core objects from first 2,000 audited events. BPI2012 check: 2,034 applications, 42,366 events, denormalized requested amount clearly labeled. Each has native/declared-denormalized and structural views, clean plus three mixed/pipeline fault seeds (14 tasks/domain). No repair result was used to choose cohorts. Final repair runs are now starting.

The core runner is scripts/run_confirmation_suite.py. Suites: primary (six paired original/varied jobs/domain), clean, pipeline, source, adaptation, structural, controls. It uses bounded CPU concurrency, process locks, per-round decoder checkpoints, explicit --resume, and immutable protocol/model checks. Complete every declared comparison; retain failures. Target-local adaptation accepts only the task observation; its fixed script/source/cache/calibration provenance is verified before evaluation. No original clean reference enters its fitting.

Four reserved public simulation files (ForgeFlow/PermitFlow/ProcureChange/TrialVersion, DOI 10.5281/zenodo.22079246, CC BY 4.0) remain unavailable: API/direct-file downloads timed out even with approved network and IPv4. Metadata/checksums are local; no records from those four were decoded. Retry sensibly while local confirmation proceeds; do not call a download failure a model result. Data root data/external/relational_ocel still contains only the two development logs and metadata.

96 held-out composition worlds have been generated as workflow_confirmation_v1 (seed 470000, test split). Paired attribute generation workflow_confirmation_attributes_v1 (seed 1321000) is running. After it completes, evaluate all six paired original/varied models with scripts/evaluate_edit_policy.py --confirmation-protocol docs/CONFIRMATION_PROTOCOL_V1.json --generated workflow_confirmation_attributes_v1 --episodes 96 (rounds default 8, budgets16/32). The original v2 model needs --calibration edit_feature_v2_attributes_calibration_v3. These are within-simulator compositional tests, not independent domains.

## Remaining work

Finish all frozen real-source and external-source comparisons, generated composition tests, candidate coverage, application case/failure analysis, uncertainty across actual trials, final figures and manuscript rewrite. Complete the published-method comparability review: HERE is a closed-case-vocabulary missing-case method; SANAGRAPH assumes supplied traces; the pinned EKG implementation needs physical load/unload/location/batch semantics absent here. Do not label task adaptations as exact reproductions. New EKG code audit is in PUBLISHED_BASELINE_CONTRACT.md. No broad SOTA claim is supported.

paper/main.tex/main.pdf are internal development drafts and compile using the copied local cache; no publication-readiness claim. Final package needs reproducible commands/artifacts, bibliography and claim-to-evidence audit. User final review only, no external submission. The goal is not complete for software or a draft alone.

## Operational details

Python /home/zubair/zubairEnv/bin/python. Research root /mnt/zubair/Temporal_Graph_Repair. Most scoring is CPU. Scoped GPU approval exists for scripts/train_edit_policy.py; use a direct command without shell redirection to reuse it. Respect other GPU users. No environment installs needed. 41 research tests and 10 tests/test_temporal_factory.py tests pass. Current frozen code check and diff check pass.

Read STATE.json for job handles; inspect logs/reports before starting anything. Whole-graph results checkpoint in data/decoding/<output>/checkpoint.json; --resume requires exact implementation/input/model pins. Preserve interrupted runs rather than overwriting. Factory edits are limited to the existing temporal_factory namespace; general world_factory/RF/wheel work is unrelated and concurrent. Do not commit the entire factory repo.

Prior evidence/corrections remain in ITERATION_01..05 and LEDGER. Corrected external exception-query audit uses native flags, not hidden learner attributes. Source first/last metrics mean timestamp correctness, not exact boundary event recovery. The valid matcher optimization is v2 (5.3544→1.3253 seconds identical retrieval/features); v1 timing scopes differed. Source annotation 16/64/256 comparisons use extra source calibration labels and are not total-label budgets.
