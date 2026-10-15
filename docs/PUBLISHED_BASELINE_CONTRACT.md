# Published methods and compatible controls

Directly copying a published score into our table would compare different tasks and information. Keep native-method results, adaptations, and generic controls separately named.

## HERE

Author repository: https://github.com/FlZtz/Hybrid_Elusive_Case_Repair_Method, pinned commit `814b1db618aaa9f14e6f471504699ade9fcbf203`; retrieved model, dataset, configuration, training source and MIT license are in cache/literature. The training implementation constructs a target tokenizer over observed case-ID labels and makes random train/validation splits over its sequences. Its native output is a missing single case identifier; it has no native alias-merge or multi-object edit output. Closed-vocabulary case prediction cannot silently become an inductive cross-schema baseline. An unchanged native-task experiment would require a separate explicitly transductive missing-case track. No such reproduction result is claimed yet.

## SANAGRAPH and heterogeneous GraphSAGE

The inspected author preprint (https://arxiv.org/html/2508.05145v1) uses heterogeneous SAGE convolutions over attribute nodes within supplied traces. Our task has unreliable trace membership. A compatible control therefore uses relation-specific mean GraphSAGE aggregation over the observed event/entity graph, with the same candidate pools and edit heads as the other neural controls. It changes the input graph, target, features and training regime. Name it **heterogeneous GraphSAGE edit control**, or explicitly **SANAGRAPH-inspired adaptation**; never label its scores an exact SANAGRAPH reproduction. Final chapter access remains incomplete, as recorded in LITERATURE_REFRESH.

The implementation uses a root projection plus a separate mean-neighbor projection for each observed relation type. It does not use edge-time features in aggregation; scalar temporal node/action inputs are shared with matched controls. A structural-only ablation would additionally need to remove those temporal inputs. Report the architecture and information boundary explicitly.

## OCIP

The pinned author code contains object-clone, lost-memory and false-memory injections and pattern detectors. The exact attribute-set clone comparison already implemented is a task adaptation to the available descriptive fields, not the original six-field detector reproduced on its native dataset. Our independent pipeline faults are separate declared mechanisms. Neither their names nor the shared use of an OCEL imply a native-method reproduction.

## Strong controls retained

Retain fixed temporal/context matching; a source-trained participant matcher; learned source/synthetic identity matchers; the refreshed source sequential pipeline; matched-candidate boosted edits, MLP, temporal message passing and heterogeneous GraphSAGE; one-pass and fixed-sequence decoders; and a no-repair control. Synthetic-only, source-only, mixed, and any future pseudo-label adaptation must have separate rows and label budgets.

## Event Knowledge Graph identifier inference

Swevels, Dijkman and Fahland, *Inferring Missing Entity Identifiers from Context Using Event Knowledge Graphs*, BPM 2023, pp. 180–197, DOI https://doi.org/10.1007/978-3-031-41620-0_11. Author institution: https://research.tue.nl/en/publications/inferring-missing-entity-identifiers-from-context-using-event-kno/. The method uses physical/contextual knowledge and five inference rules for missing identifiers; it already establishes practical graph-based history reconstruction. Our paper cannot claim that problem or the use of contextual event graphs as new.

Author repository https://github.com/Ava-S/ekg_inference, archived snapshot `3af6ab2a2d1406175b0075672cf8b9fe6cc9f90f`, uses a Neo4j-based implementation with activity/location context. A faithful baseline requires its declared physical constraints; our generic observations do not supply all of them. Do not equate a nearest-context heuristic with an exact reproduction. The extended author manuscript is at https://zenodo.org/records/7802242; the first PDF transfer failed, so detailed rule reconciliation remains pending.

## Existing executable fault and repair languages

FLAWD already specifies and injects event-log imperfections to construct labeled benchmarks: Comuzzi, Ko and Maggi, *A Language to Model and Simulate Data Quality Issues in Process Mining*, JDIQ 17(2), 6, 2025, https://doi.org/10.1145/3743144; author code https://github.com/jonghyeonk/FLAWD. Generic controllable fault injection is not a contribution of this paper.

Sadeghianasl et al., *A Language for Efficient Event Log Repair*, EDOC 2026, provides algebraic repair expressions: https://conf.researchr.org/details/edoc-2026/edoc-2026-papers/8/A-Language-for-Efficient-Event-Log-Repair. An auditable edit vocabulary alone is likewise insufficient novelty. Our study must establish the incremental application value of generated training under scarce verified histories and coupled association/identity errors.
