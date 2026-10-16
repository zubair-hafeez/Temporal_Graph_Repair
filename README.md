# Generative temporal graph repair

Active research project: learn transferable restoration of event participation and fragmented identities from generated operational histories. The intended application is traceability in purchasing, loan processing, and logistics.

The accepted [research proposal](../Temporal_Graph_Papers_2024-2026/analysis/new_idea_20261015/PROPOSAL.md) and [novelty audit](../Temporal_Graph_Papers_2024-2026/analysis/new_idea_20261015/NOVELTY_AUDIT.md) are the starting point. Results, including failed hypotheses, determine the final claims. Development experiments found both useful transfer and consequential failures. The [confirmation protocol](docs/CONFIRMATION_PROTOCOL_V1.json) is now frozen and its experiments are running. The manuscript remains an internal draft until those comparisons and the final review are complete.

Read [RESUME.md](RESUME.md), [STATE.json](STATE.json), and [docs/TASK_CONTRACT.md](docs/TASK_CONTRACT.md) before continuing. The completed forecasting paper remains a separate project. Its raw public data can be read; its source, labels, results, and release artifacts must remain unchanged.

The implemented factory component separates executable workflow histories, observation variation, recording faults and task-specific exports. It supports this paper's repair study and a bounded future-participation interface check. The complete general dataset factory is outside this paper's scope. RF is excluded from the experiments.

Interpreter: `/home/zubair/zubairEnv/bin/python`. Commands are run from this project directory. Generated datasets, checkpoints, and bulky downloaded artifacts belong outside version control.

The strongest current generated-data learner uses boosted edit scoring. Neural controls and negative results are retained; this project does not claim a new neural architecture or state-of-the-art performance against incompatible published tasks. [Published comparison boundaries](docs/PUBLISHED_BASELINE_CONTRACT.md) state which methods assume correct trace containers, a closed case vocabulary, or additional physical semantics.

The analysis command below validates completed report/task/protocol hashes and exports per-trial and per-type CSV files. It marks unfinished groups explicitly and does not average an incomplete primary comparison:

```bash
python scripts/collect_confirmation_evidence.py \
  --domains bpi2017 bpi2019 bpi2012_check \
  --output confirmation_real_evidence_v1
```

The seven evaluation suites are `primary`, `clean`, `pipeline`, `source`, `adaptation`, `structural`, and `controls`. `scripts/run_confirmation_suite.py --plan-only` lists exact jobs without running them. Existing runs require `--resume`; process locks prevent duplicate suite execution. Frozen files must remain unchanged during confirmation. The [operational continuation declaration](docs/CONFIRMATION_OPERATIONAL_CONTINUATION_V1.json) records recovery after interrupted tool processes, preserving partial reports and the original scientific choices.
