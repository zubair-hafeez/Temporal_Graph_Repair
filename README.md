> **Complete research snapshot:** Code and manuscript files are browsable here. Datasets, checkpoints, caches, ignored files and original local Git metadata are included in the split archives attached to [release research-snapshot-20261107T032034Z](https://github.com/zubair-hafeez/Temporal_Graph_Repair/releases/tag/research-snapshot-20261107T032034Z). See [restore instructions](github_snapshot/RESTORE.md) and [verification report](github_snapshot/VERIFICATION_REPORT.md). A normal Git clone does not download those archives.

PAPER RELOCATED
Compiling Temporal Walks for Exact Counterfactual Audits now lives in
/mnt/zubair/Temporal_Walk_Counterfactual_Audits/
Its PDF is PAPER.pdf there. The local rEPAIR. LATEST_PAPER.pdf
opens this directory's earlier workflow-repair paper.

# Temporal graph learning and repair

**Latest paper:** [Compiling Temporal Walks for Exact Counterfactual Audits](/mnt/zubair/Temporal_Walk_Counterfactual_Audits/paper/main.pdf) — **12 pages including references and appendices**. Its shortcut is `/mnt/zubair/Temporal_Walk_Counterfactual_Audits/PAPER.pdf`.

The new paper develops exact repeated deletion evaluation for bounded-degree temporal walk predictors. It includes proofs, a compact endpoint-transfer representation, optimized exact controls, and a protocol frozen before later confirmation outcomes were opened.

Across three datasets, confirmation covers **6,000 predictions and 113,644 deletion hypotheses in 72 audits**. Median total-cost speedups for dispersed historical candidates are **20.30×, 4.32×, and 38.81×** on MOOC, Wikipedia, and LastFM. Recent-source candidates include losses. The independent-edit approximation matches every observed confirmation decision; the paper reports that limitation explicitly.

**46 tests pass.** A separate neural-decoder development replication adds nine fits and 40,037 hypotheses. Saved prediction ranks, candidate cohorts, witnesses and timing arithmetic were independently checked; the full manuscript was visually reviewed. These results support a specialized computational contribution, not general predictive SOTA or a guarantee of conference acceptance.

Start with the [research overview](/mnt/zubair/Temporal_Walk_Counterfactual_Audits/docs/RESEARCH_HISTORY.txt), [reproduction guide](/mnt/zubair/Temporal_Walk_Counterfactual_Audits/docs/REPRODUCE.txt), [claim boundaries](/mnt/zubair/Temporal_Walk_Counterfactual_Audits/docs/CLAIMS.txt), and [final review](/mnt/zubair/Temporal_Walk_Counterfactual_Audits/results/final_review.json). Editable manuscript: [main.tex](/mnt/zubair/Temporal_Walk_Counterfactual_Audits/paper/main.tex).

The [earlier workflow-repair study](revision/paper/main.pdf) is a distinct completed study. Its equal-budget generation comparisons and never-merge baseline remain preserved in [confirmation_v3](revision/confirmation_v3/RESULTS.txt). Generation helped conditionally at a small loan-data budget, but did not establish general benefit and worsened the tested larger-budget settings. Natural-error adjudication and human-time collection remain unmeasured, as authorized.

## Historical workflow-repair documentation

The material below describes the earlier study; current-paper instructions above take precedence.

Research manuscript and reproduction package for author review: learn transferable restoration of event participation and fragmented identities from generated operational histories. The application is traceability in purchasing, loan processing, and logistics. Start with [the latest paper](revision/paper/main.pdf) and [the plain-English review guide](REVIEW_NOTES.txt).

The archived [research proposal](docs/proposal_archive/PROPOSAL.md) and [novelty audit](docs/proposal_archive/NOVELTY_AUDIT.md) record the starting point. All 84 original comparisons on the three available operational cohorts, six generated-model evaluations and nine separately declared calibration-sensitivity runs are complete. The [completion record](docs/CONFIRMATION_COMPLETION_SCOPE_V1.json) accounts for four additional predeclared simulation sources that could not be retrieved; their comparisons were not executed. The [original protocol](docs/CONFIRMATION_PROTOCOL_V1.json) and negative results remain unchanged.

Read [RESUME.md](RESUME.md), [STATE.json](STATE.json), and [docs/TASK_CONTRACT.md](docs/TASK_CONTRACT.md) before continuing. The completed forecasting paper remains a separate project. Its raw public data can be read; its source, labels, results, and release artifacts must remain unchanged.

The implemented factory component separates executable workflow histories, observation variation, recording faults and task-specific exports. It supports this paper's repair study and a bounded future-participation interface check. The complete general dataset factory is outside this paper's scope. RF is excluded from the experiments.

Interpreter: `/home/zubair/zubairEnv/bin/python`. Commands are run from this project directory. Generated datasets, checkpoints, and bulky downloaded artifacts belong outside version control.

The principal learner, selected on development results, uses boosted edit scoring. Neural controls and negative results are retained; this project does not claim a new neural architecture or state-of-the-art performance against incompatible published tasks. [Published comparison boundaries](docs/PUBLISHED_BASELINE_CONTRACT.md) state which methods assume correct trace containers, a closed case vocabulary, or additional physical semantics.

The analysis command below validates completed report/task/protocol hashes and exports per-trial and per-type CSV files. It marks unfinished groups explicitly and does not average an incomplete primary comparison:

```bash
python scripts/collect_confirmation_evidence.py \
  --domains bpi2017 bpi2019 bpi2012_check \
  --output confirmation_real_evidence_v1 --require-complete
```

The seven evaluation suites are `primary`, `clean`, `pipeline`, `source`, `adaptation`, `structural`, and `controls`. `scripts/run_confirmation_suite.py --plan-only` lists exact jobs without running them. Existing runs require `--resume`; process locks prevent duplicate suite execution. Frozen files must remain unchanged during confirmation. The [operational continuation declaration](docs/CONFIRMATION_OPERATIONAL_CONTINUATION_V1.json) records recovery after interrupted tool processes, preserving partial reports and the original scientific choices.

## Earlier consolidated-project workflows

Use [the current reproduction guide](revision/confirmation_v3/REPRODUCE.txt) for the new confirmation and manuscript. Older review manifests and consolidation records concern their historical artifact states; they are retained for provenance. The documented environment is Python 3.10.

### Historical original-package instructions

The instructions below describe the earlier package layout. Its separate folder was retired during consolidation; its manifests remain under `archives/review_manifests/`. They are retained as historical instructions, not directions to find an existing separate review folder.

The local review bundle preserves the sibling layout `Temporal_Graph_Repair/` and `RF_Dataset_Factory/`. Only the bounded temporal generator and generic storage helpers are copied from the latter. No RF experiment or dataset is included. Run commands from the bundled `Temporal_Graph_Repair` directory, using Python 3.10 and the versions in `requirements.txt`. The tested environment already contains these packages; installation is needed only in a new environment.

First verify the copied files, then run the observation-boundary, decoder and factory checks:

```bash
python scripts/verify_review_bundle.py --bundle ..
python -m pytest -q tests ../RF_Dataset_Factory/tests/test_temporal_factory.py
```

The bundle includes the full 96-world generated confirmation collection, frozen model weights, validation-score caches, saved edit programs and reports. A small CPU replay independently reconstructs the first recorded graph for each supplied run and checks both its complete application metrics and its exact edit program. Use a new output name on every invocation:

```bash
python scripts/audit_generated_replay.py \
  --reports confirm_generated_plain_r0 confirm_generated_varied_r0 \
            confirm_generated_plain_r1 confirm_generated_varied_r1 \
            confirm_generated_plain_r2 confirm_generated_varied_r2 \
  --output reviewer_sample_replay
python scripts/collect_generated_confirmation.py --output reviewer_generated_summary
```

The full generated evaluation is reproducible with `scripts/evaluate_edit_policy.py`, using the model, calibration and other arguments recorded in each original report's `config`. Supply a new `--output` so historical evidence remains intact. The generated test uses eight passes and candidate budgets 16/32. Operational-source tests use four passes and budgets 128/64; they are different declared regimes.

Third-party operational source files and their derived task graphs are omitted from the review bundle. Their exact SHA-256 hashes, adapter options and cohort exclusions are in `docs/CONFIRMATION_PROTOCOL_V1.json`; dataset citations are in the manuscript. Obtain the original files and create a JSON object mapping source domain names to their local paths. A path-only protocol copy checks every source hash and records its relationship to the original freeze:

```bash
python scripts/make_reproduction_protocol.py \
  --source-map /absolute/path/to/source_paths.json \
  --output docs/REPRODUCTION_PROTOCOL.json
python scripts/prepare_confirmation_tasks.py \
  --protocol docs/REPRODUCTION_PROTOCOL.json --domain bpi2017
python scripts/evaluate_external_policy.py \
  --task data/reproduction_confirmation_v1/bpi2017/native/mixed_91001 \
  --model edit_feature_attributes_v3 --kind feature \
  --calibration edit_feature_attributes_v3 \
  --protocol docs/REPRODUCTION_PROTOCOL.json --output reviewer_bpi2017_r0
```

Run this in a disposable copy of the review bundle: task preparation writes its own preparation report under the historical report name. The path-only protocol preserves model, calibration, cohort, source checksum, fault and decoder settings. Its historical freeze date is copied for provenance and does not represent a new prospective registration. BPI 2012 check is a derived split. Reconstruct it from the exact original BPI 2012 input before using the derived graph in the source-path map:

```bash
python scripts/rebuild_bpi2012_check.py \
  --source /absolute/path/to/bpi2012.xes.gz \
  --output /tmp/rebuilt_bpi2012_check_attributes.json.gz
```

This helper validates the raw-source hash, uses the unchanged frozen adapter, and requires the derived bytes to match the original check-split hash. It does not require the unrelated logistics development file or the original project's directory layout.

Training recipes are retained under `data/recipe_manifests/`, and the training/generation commands and source snapshots are in the original reports. Those manifests are not full training caches. Recreate worlds and edit-learning states in a separate working copy before retraining. Frozen-model replay and independent retraining answer different reproducibility questions; retrained binary weights need not be identical across numerical-library versions or hardware.

For a fresh execution of an entire paired training recipe, create a code-only workspace with empty result directories. The helper refuses an existing destination and never deletes or replaces historical artifacts:

```bash
python scripts/prepare_training_workspace.py --output-dir /tmp/tgrepair_retrain
cd /tmp/tgrepair_retrain/Temporal_Graph_Repair
python scripts/run_training_replicate.py --replicate 1
```

This generates 512 training and 96 validation worlds, creates their paired observation variants and edit-learning states, and fits both original-field and varied-field models. It can take substantial time and disk space; its recorded stages support `--resume` after completed stages. Replicate 2 has its own fixed seeds. The original replicate-zero commands and artifact names are retained in its individual generation, training and calibration reports.

The artifact is prepared for private final review. It does not submit the manuscript, publish data, choose author affiliations, or grant a new software license. Third-party dataset terms continue to apply.

The [source-archive audit](reports/source_snapshot_audit_v1.json) verifies all 60 historical source archives against their per-file inventories. Eight original reports captured a transient archive checksum during a concurrent write; the audit preserves those fields and records the correct complete-archive checksums separately. Model, calibration and dataset pins are unaffected. The archive writer now publishes files atomically, and the local suite passes 52 checks including a concurrent-write regression. The 28 frozen inference/adapter files remain unchanged.

The manuscript PDF is included. Its LaTeX source uses standard article, AMS math, graphics, booktabs and hyperref packages, with Latin Modern extension symbols. From the `paper` directory, `tectonic main.tex` builds it when the TeX bundle is available. The working study uses a local cached Tectonic installation; no TeX binary or complete package installation is assumed to be part of the review bundle. Figures are supplied as vector PDFs, with their generating scripts and source hashes.

`python scripts/build_manuscript.py --compiler tectonic` also records the exact source/PDF correspondence. `python scripts/audit_final_review.py` checks the completed evidence, manuscript dependencies and reproduction records without rerunning model experiments. The evidence audit requires original operational task files only if you choose to regenerate the aggregate analysis; the supplied aggregate and its source-report hashes are included in the review artifact.
