# Resume temporal graph repair research

User authorized autonomous work through final review. The active goal remains incomplete; do not submit externally. Main BPI 2017/2019 tests and BPI 2012 check remain sealed. Preserve the prior paper and all negative results.

Read STATE.json and docs/ITERATION_03.md. Source-trained identity matching now raises development history F1 to 0.9107 after participant repair (0.8922 without merging). Original synthetic identity matching reaches only 0.8959. Generated histories were too short; the bounded factory TR1 extension adds routes, repetition, lifecycle records and waiting variation. Expanded train/validation collections v4 are generated and source-pinned. Their performance is not yet established.

First synthetic edit-policy experiment: temporal neural action macro-AP 0.7398, MLP 0.7356, boosted features 0.7701. On 24 generated validation worlds at validation-selected 95% action precision: mixed-history F1 0.7625 before repair, 0.8364 temporal, 0.8330 boosted, 0.8245 MLP. Clean F1 is 0.99981 temporal and 1.0 boosted. These small development differences do not establish a publishable model contribution. Fixed-sequence boosted decoding is 0.8319.

Active subsequent experiments: edit_train_v2 preparation and identity_learned_synthetic_v2 use workflow_train_v4. Check their reports/logs before restarting. edit_validation_v2 is complete. Original GPU models have finished; GPU access is approved narrowly for scripts/train_edit_policy.py. Respect other users' GPU processes. The original sandbox CUDA failure is preserved in reports/edit_temporal_v1.log.

Next: finish expanded-generator checks, train matched controls, assess whole-history outcomes and source transfer, then decide whether coordinated repair/generative training has sufficient application value. Full final experiments, independent faults, published-method adaptations, manuscript and review package remain outstanding.

The matching attribute cache has an exact 51,200-row regression against the prior implementation, with feature time reduced from 5.31s to 1.15s in that check. Further tests are recorded separately. Use source snapshots for every substantial run. Preserve existing early exploratory evidence and do not retrospectively call it frozen evaluation.

Factory scope remains src/temporal_factory and its dedicated tests. Another session is organizing the general factory; preserve all concurrent changes and merge only owned fields in shared planning files. General world_factory, RF, packaging and the previous research paper are outside this experiment. Source imports use the existing Python 3.10 environment; root factory wheel installation is not validated.
