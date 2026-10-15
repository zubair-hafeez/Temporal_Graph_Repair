# Resume generative temporal graph repair

User authorized autonomous paper development and experiments through final review. Goal remains active; no submission or publication is authorized. The proposed neural method and publication readiness are not established.

Read STATE.json, docs/TASK_CONTRACT.md, docs/ITERATION_01.md, and docs/ITERATION_02.md. Main BPI 2017 and BPI 2019 targets remain sealed. BPI 2012 and logistics are development only. The prior forecasting paper is unchanged.

Completed: anonymized source adapters; nonleaking observation faults; strong fixed/feature matching; known-hole and unknown-location mixed-error probes; whole-graph sequential baseline; initial identity controls; bounded integrated factory workflow and repair/prediction plugins. Fifteen research tests and nine scoped factory tests passed.

Key results: with amount attributes, the learned matcher reaches 97.3% independent-gap and 82.5% contiguous-gap recall on BPI 2012 tune. Whole-graph participation repair raises history F1 0.7445 to 0.8922, while the simple identity-merging extension reduces it. See reports/sequential_development_v1.json. The identity baseline is not yet strong enough to establish the proposed joint-learning advantage.

Factory acceptance: ../RF_Dataset_Factory/reports/temporal_repair_acceptance.json. New source namespace is temporal_factory, with research task plugin tgrepair/factory_plugin.py. The complete general factory and RF work remain outside this paper. Existing factory dirty changes are preserved. Pilot v1/v2 are development only; v2 coupled flat projection to service vocabulary. Correct the training schedule to vary schema, projection, and mechanism composition independently before training.

Next exact work: define disjoint training/validation mechanism compositions, generate a bounded source-pinned training set with clean/mixed variants, implement learned identity and sequential controls, then train a compact schema-conditioned temporal graph edit model. Use source snapshots for every substantial run and preserve failed experiments. Source snapshots live under data/source_snapshots/ and are referenced in reports. Early exploratory baselines have hashes/caches but no complete historical source archives; do not treat them as frozen publication experiments.

Interpreter: /home/zubair/zubairEnv/bin/python. Two A6000 GPUs are present; no GPU training has started. Respect existing processes. Root factory packaging declares Python >=3.12; the new research source has been checked under the installed Python 3.10 environment, not as an installed root wheel.

No background job remains required from the current baseline/generator pilots; verify session state before assuming any later training is complete.
