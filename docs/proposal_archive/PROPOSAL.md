# Generative Temporal Graph Repair for Operational Traceability

Research proposal, 15 October 2026.

**Recommended direction:** learn to reconstruct connected operational histories when event relationships and object identities have been damaged. Train the model on generated workflows with known histories, then test whether it can repair real purchasing and loan-processing records from a new process. The application is traceability: establishing what happened to which order, item, offer, or resource, and preserving genuine exceptions in that history.

This is a new project alongside the completed forecasting paper. The present evidence establishes a practical problem, relevant data, strong competitors, and a feasible experiment plan. It does not establish a successful learned model. The detailed prior-work assessment is in [the novelty audit](NOVELTY_AUDIT.md).

**Governing platform requirement:** the long-term system is a general scientific temporal graph dataset factory for researchers across domains and tasks. This paper must contribute reusable capabilities to that factory through shared interfaces and an integrated workflow module. RF remains an optional domain module outside this paper. The existing [temporal-world framework proposal](../../../RF_Dataset_Factory/docs/TEMPORAL_WORLD_FRAMEWORK_PROPOSAL.md) supplies the broader architecture; the work below is an incremental implementation and research validation of part of it. Broad extensibility is a design requirement, while demonstrated support must be stated separately for each implemented module and task.

## The application problem

Consider a purchasing system in which an invoice-receipt event is attached to the wrong order item. Another item has acquired two identifiers during an import, while some receipt-to-item links are missing. A downstream analyst can now flag the wrong purchase as unpaid or fail to find the goods associated with an invoice. Repairing each cell independently can produce individually plausible answers that disagree about the same item's history.

There is a second difficulty: an unusual sequence may be real. A cancellation, repeated approval, partial delivery, or invoice received before goods receipt must retain its place in the relevant history. A repair system that converts every unusual execution into a typical one can make the data less useful.

The practical objective is therefore:

> Given an imperfect event graph and the process schema, restore supported object identities and relationships, retain the actual behavior, and identify unresolved parts for review.

The application is broad enough to cover purchasing, financial-service workflows, logistics, and later software operations. The first paper should establish the capability on two real process families and a controlled logistics setting. It should not promise universal recovery across all domains.

## The research gap

Event-log repair is established research. SANAGRAPH already uses heterogeneous GNNs for attribute restoration; HERE repairs missing case identifiers; unified LLM repair handles several imperfections; and object-link reconstruction implementations exist. Those are competitors, not motivation to claim an empty field. See [SANAGRAPH](https://arxiv.org/abs/2508.05145), [HERE](https://link.springer.com/article/10.1007/s12599-025-00935-5), and the [unified LLM repair dissertation](https://epub.uni-bayreuth.de/id/eprint/8484/1/Dissertation_Sebastian_Schmid.pdf).

The proposed gap concerns **restoring the connected structure of an unfamiliar workflow when neither its event grouping nor its object identities are fully reliable**. The central research question is:

> Can generated workflow histories teach a temporal graph model reusable repair operations that transfer to a new process without clean training histories from that target process?

The strongest version also preserves which objects experienced genuine exceptions. Transfer, coordinated repair, and useful application outcomes must all be demonstrated. “GNN plus LLM plus constraints” is not a sufficient contribution.

## Why a temporal graph is appropriate

An event can involve several entities: an item, its purchase order, a vendor, and a resource. Different events connect overlapping subsets of those entities. Their ordering provides evidence about which history a damaged record belongs to.

Use event nodes and typed object nodes. Event-object edges record participation; object-object edges are included only where the source actually supplies them or an explicit source field justifies their derivation. Temporal edges connect observed events associated with the same object. Edge construction must be repeated after corruption so that a hidden relation cannot survive indirectly as an uncorrupted temporal edge.

This is a historical restoration task. Its default input is a fixed observation window, and evidence on either side of a damaged event is allowed inside that window. An online prefix-only setting would be a separate experiment. Calling offline repair “forecasting” would misdescribe the information available.

A graph method has to earn its place. Compare it with a sequence Transformer using the same event attributes and context, a static graph, and strong temporal matching. If those alternatives match it, the paper cannot claim that temporal message passing is responsible for success.

## Dataset selection

| Dataset | Role | Available evidence and important limits |
| --- | --- | --- |
| **BPI Challenge 2019** | Main real-source application: purchasing traceability | Public purchasing records. The downloaded OCEL conversion has purchase-order, item, resource, and vendor types. A bounded audit inspected 2,000 events and 7,470 participant links. The conversion supplies no object-object links or changing object attributes. |
| **BPI Challenge 2017** | Second real-source application: application and offer histories | The existing raw XES file is available locally. In the first 500 applications, the audit found 20,295 events and 768 explicitly identified offers. Use applications, offers, and recorded resources; do not invent offer associations for events that lack them. |
| **BPI Challenge 2012** | Real-source development only | Its raw XES and metadata already exist locally. Use a declared subset for feasibility and validation. It is the earlier system of the same financial institution represented in BPI 2017, so that transfer must not be described as transfer to an unseen organization. |
| **Container Logistics OCEL** | Development and independent simulator reference | The downloaded version has 35,372 events, 13,882 objects, seven object types, and 14 event types. It supplies interactions between orders, handling units, containers, documents, and transport resources. This is simulated data. |
| **New generated workflow collection** | Main source of paired training examples and controlled generalization tests | Generate complete histories, corrupted observations, and repair targets. Hold out whole mechanism compositions and workflow families. This collection is proposed, not already implemented. |

The [BPI 2019 challenge description](https://www.tf-pm.org/competitions-awards/bpi-challenge/2019) grounds the purchasing application, and the [original BPI 2017 record](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) describes its application and offer identifiers. The downloaded [OCEL conversions](https://zenodo.org/records/22663565) are format conversions of existing real records, not newly collected business data. The [logistics documentation](https://ocel-standard.org/event-logs/simulations/logistics/) explicitly describes simulation. File-specific checks and hashes are in [the data audit](data_feasibility.json) and [the purchasing audit](bpi2019_feasibility.json).

Keep original data and its attribution. The newly downloaded Zenodo records declare CC BY 4.0; the existing raw BPI 2012 and 2017 metadata instead name the 4TU General Terms of Use. Preserve those source-specific terms and distribute acquisition scripts with the research code. Cite the original BPI dataset and the OCEL transformation study when using the conversion. The purchasing archive is about 88 MB compressed but 3.48 GB uncompressed; stream it into a compact typed representation instead of loading multiple full copies into memory.

**Reference truth:** real records supply the reference values for controlled corruption. They are not certified clean histories. Score only deliberately changed, originally observed facts. Preserve naturally unusual records in the reference. No result from this protocol alone supports a claim that naturally occurring business-data errors have been correctly repaired.

Angular GitHub commits were considered and excluded from the main evaluation. The file associates its earliest 2014 commit with 119 branches, which may reflect later branch membership. Those relations need valid-time semantics before serving as historical ground truth.

## The proposed learning task

The input is an observed temporal event graph, a schema description, and a repair policy specifying allowed edits. The output is a compact edit program and its resulting graph. The model may also return unresolved candidate alternatives.

The first paper should jointly cover three common structural problems:

1. Missing event-object relationships, including correlated gaps affecting one object or a sequence of observations.
2. Incorrect event-object relationships, including assignments to a plausible competing object of the same type.
3. Fragmented identity, where one real object appears under multiple identifiers during its history.

Coarse or tied timestamps can be an input condition. Recover a supported partial ordering where possible; exact timestamp regression is not the core objective. Fully missing event invention, arbitrary attribute repair, and separating several physical objects collapsed into one identifier are later extensions. The substantive first-paper problem is restoration of interacting histories across processes, rather than one particular missing field.

The deployment task does not reveal which links are wrong, how many links are missing, or which identifiers are aliases. Include a simpler known-hole track for comparison with published methods, but keep it separate from the main diagnosis-and-repair track.

Alongside artificial graph masking, implement an independent ingestion-failure test: omit join rows from a bounded export batch, change an identifier namespace partway through a history, and apply a stale participant mapping to later records. These executable faults provide concrete causes for damaged relationships. Apply each fault consistently to every affected field, then compare with the source record. This is controlled pipeline failure on real-source data, not a naturally observed incident. Hold out some fault implementations and combinations from training.

## Generating useful training data

Build a bounded stateful workflow module within the factory's general interfaces. Keep its execution rules separate from observation policies, task builders, and export adapters so that the same generated history can support several temporal graph learning tasks. Its first reusable mechanisms should be:

| Mechanism | Example of an execution | What repair could learn |
| --- | --- | --- |
| Branching and joining | Several offers or parallel approvals for one request | Shared context does not imply identical objects. |
| Batching and splitting | Several items enter one container, then separate | Event participation can be many-to-many and changes over time. |
| Resource handoff | Work passes between people or machines | A resource connects different histories without merging their identities. |
| Rework and retry | A failed check is repeated | Repetition can be genuine behavior. |
| Cancellation and return | An item changes course after an earlier commitment | An unusual transition must remain attached to the right object. |
| Shared timing and queues | Several tasks wait for a common resource | Temporal proximity helps but is not sufficient evidence of identity. |

Separate **execution variation** from **observation corruption**. First execute a complete workflow, including normal variation and genuine exceptions. Then apply an observation program that drops links, assigns a wrong participant, changes an identifier for part of a history, or coarsens timestamps. The target is the executed history, including its exceptions.

Use several simulated observation streams where the recipe supports them, but do not pretend that the real BPI files provide independently verified multi-system ground truth. For those files, evaluation uses declared artificial corruption of available fields.

Each generated episode should contain:

- A recipe, mechanism ancestry, generator version, and separate execution and corruption seeds.
- The full reference history in an evaluator-only artifact.
- The observed graph and allowed schema information in the learner artifact.
- The corruption program, affected reference facts, and acceptable repair equivalences in evaluator-only labels.
- Traceability questions and answers, including which objects experienced an exception.

Build several valid observation variants from the same history, including an uncorrupted variant. Keep all variants of a history in one split. This supplies both repair supervision and examples in which the correct action is to leave the record unchanged.

Start with roughly 2,000 small episodes for debugging and 20,000 for the first learning comparison. These are proposed budget limits, not claims that this data volume will be sufficient. Vary topology and mechanism composition, not just random seeds or entity names. Compare stateful generation with random typed graphs and with corruption of available real training logs at matched event counts.

Generic synthetic pretraining is already established by [PluRel](https://arxiv.org/abs/2602.04029). The proposed contribution is the value of executable histories and their observation programs for this restoration task.

## A feasible model design

Use a compact temporal graph encoder followed by an autoregressive edit decoder. This gives generation a concrete purpose: producing coordinated changes to the graph.

**Encode the evidence.** Embed event types, object types, permitted attributes, relative time, and observed neighborhoods. Start with two or three message-passing layers and bounded local histories. Use opaque identifiers consistently within an episode; do not learn a classifier over global order or offer IDs. A lightweight text encoder may represent schema descriptions, but it is not required to generate the event records.

**Represent the schema as a prompt.** A structured prompt specifies object and event types, relation roles, optional cardinality information, and permitted edits. For example, it can state that the target is event participation and that recorded activity names are protected. The initial method is generative learning with schema conditioning. Call it learned prompt adaptation only if a distinct learned prompt mechanism is implemented and shown to help.

**Retrieve candidates.** For each proposed edit, retrieve same-type objects from observed temporal neighborhoods and available attribute context, plus an unresolved option. Measure candidate recall separately. If the correct object is absent, no decoder can repair it. Use the same candidate pool for the main neural and heuristic comparisons; also measure end-to-end retrieval performance.

**Generate coordinated edits.** A minimal edit vocabulary is `ADD_PARTICIPANT`, `REMOVE_PARTICIPANT`, `MERGE_ALIASES`, `STOP`, and `UNRESOLVED`. Arguments point to existing events and object handles. After a merge or relation edit, update the local graph before the next decision. This allows identity and relationship decisions to inform each other. Store the output as a reversible patch; the source data remain available.

**Check the patch.** Enforce referential integrity, type compatibility, allowed operations, and explicitly protected evidence. Use business-process expectations as context or soft evidence. Do not force every history to comply with a normative process: a genuine violation can be exactly what an analyst needs to see.

**Train.** Supervise edit decisions and final graph reconstruction. Add uncorrupted and genuine-exception episodes whose correct changes are empty or localized. A practical starting loss combines edit prediction, participant selection, alias matching, and a penalty for changing unchanged facts. Use one canonical training edit order, but evaluate the reconstructed graph modulo equivalent object renaming and edit order. Two valid patches should not be treated as different scientific answers merely because their syntax differs.

**Handle ambiguity.** Some observations support several histories. Include deliberately ambiguous small examples with enumerated alternatives. Test whether the system returns an unresolved result instead of committing to an unsupported identity. Reliable exact recovery is not possible when all distinguishing evidence has disappeared.

A 10–30 million parameter encoder and decoder is an appropriate initial engineering target. Profile memory and throughput on one GPU before scaling. The two available GPUs are useful for independent seeds and a separate LLM baseline; training a new large language model is unnecessary.

## Strong comparison methods

| Comparator | Why it is necessary |
| --- | --- |
| Exact joins and schema rules | Establish what can be repaired by available explicit evidence. |
| Role-aware co-occurrence, temporal nearest neighbors, and attribute matching | Cheap methods may solve much of the task. Include degree normalization and activity compatibility, not only raw counts. |
| Matching plus constraint optimization | Tests whether global consistency alone explains the improvement. |
| Sequential identity matching and relationship repair | Tests whether coordinated edits improve on a strong modular solution. Use an appropriate [Ditto-style matcher](https://github.com/megagonlabs/ditto) where descriptive attributes exist. |
| SANAGRAPH, HERE, and a corrected object-link GNN | Compare on the tasks each supports; clearly label adaptations. Do not grant correct case grouping to one method and then call the comparison equivalent. |
| Sequence Transformer and static graph with the same edit decoder | Isolate the value of the temporal graph representation. |
| Prompted and fine-tuned LLM repair | Tests whether serializing the same context and schema is enough. Include retrieval plus validation; a weak zero-shot prompt is not an adequate sole LLM baseline. |
| Target-process supervised model | Measures the value of having clean target examples. Report separately from methods that receive none. |

Within the zero-clean-target setting, all methods may inspect the same damaged target context and schema. Transductive matching is allowed and must be reported. Clean support examples define a separate few-example setting, with identical support budgets across eligible methods.

If a baseline implementation cannot be obtained, record the missing reproduction and implement a faithful, named adaptation from the published method. Do not quietly substitute a weaker model while retaining the original name.

## Evaluation centered on application outcomes

The primary outcome is **traceability question accuracy** on restored histories. Examples include identifying the order item associated with a receipt, retrieving an offer's application history, and finding which operational objects were affected by a recorded cancellation or rework episode. Write these queries from available schema fields and reference relations. Avoid inventing invoice entities or shipment evidence absent from a dataset.

For purchasing, a second task can compare a predefined receipt/invoice ordering audit against its answer on the uncorrupted reference. Label this as preservation of an audit result, not proof of actual financial or legal compliance. Different purchasing categories permit different event sequences.

Report the following supporting measurements:

- Participant precision and recall, and identity-clustering quality, with per-type results.
- The fraction of originally correct facts changed unnecessarily.
- Correct attribution of genuine cancellations, rework, and other selected exceptions.
- Resolved coverage and errors among committed repairs, with abstentions counted explicitly.
- End-to-end runtime, candidate recall, and memory at several graph sizes.

Use three independent training seeds for the decisive comparisons and simple uncertainty intervals over independent operational units. The scientific story should be about restoring usable records and transferring between workflows; it does not require another forecasting or calibration study.

## Splits and controls that determine credibility

**Hold out the process.** Train on generated histories and select settings on generated validation families, public simulated development logs, and a declared BPI 2012 development subset. Freeze the method before the principal BPI 2017 and BPI 2019 tests. Reserve the already inspected BPI 2017 applications and BPI 2019 sample events and their related operational objects outside the final test. Report synthetic-only training separately from source-real pretraining and target few-example adaptation. Source-real validation must be disclosed even when training weights use only synthetic examples.

Fit generator parameters and choose repair settings using source and development data. Target schema descriptions are allowed, but fitting the generator to clean target histories would invalidate the zero-clean-target claim. Record any manual target-specific rules and give the same rules to competing methods. An unseen dataset is not automatically an unseen domain; label each transfer setting precisely.

**Hold out compositions.** A new seed of a seen workflow is an in-distribution test. Reserve whole mechanism combinations, branching structures, and naming schemes for transfer evaluation. The model should not recover the identity of a generator template from an ID prefix.

**Group related operational objects.** Keep one purchase order and its items together, and one application and its offers together. Group shared events across core process objects. Vendors and resources may be shared context, but that choice does not establish generalization to unseen vendors or resources. In a chronological test, prevent held-out outcome facts from entering training context through these shared nodes.

**Remove duplicate answers.** BPI 2019 stores object identifiers both in relationships and in fields such as `cID`, `cPOID`, `cItem`, `cVendor`, and `resource`. A corrupted participation fact must be changed or hidden in all equivalent encodings, including derived edges, row ordering, and cached features. Preserve an uncorrupted observation elsewhere only when the declared scenario genuinely leaves that evidence available.

**Use the same evidence.** Prompts, rule baselines, graph models, and LLMs get the same schema, candidate catalog, and observation window. Privileged generator state, corruption masks, and repair programs stay outside learner inputs.

**Preserve difficult valid behavior.** Include clean unusual histories, real recorded cancellations, and simulated process exceptions. Compare against a normal-history-only generator. This reveals whether a model is repairing records or merely making every process look typical.

**Separate scientific effects.** The essential ablations remove temporal information, graph relationships, coordinated edits, schema conditioning, stateful generation, and exception examples. Give ablations the same candidate pool and checker so the improvement cannot be attributed only to a better search space.

The logistics file was used in exploratory probing during selection. Treat it as development evidence. Its heuristic results are not a frozen final test. Define final independent simulator and real-process partitions before model selection.

## How the dataset factory helps

The existing factory offers useful engineering patterns: canonical configuration hashing, source provenance, atomic artifact writes, bounded jobs, and group-based splitting. Its current schema, sample arrays, compiler, and dataset validator remain RF-specific. The general temporal-world document is a proposal, not an implemented general generator.

The factory is the long-term research platform. This paper should establish reusable parts of that platform and demonstrate their value through operational traceability. A standalone workflow generator with no shared integration would not satisfy that objective. Keep the common engine independent of radio dependencies; retain RF as a separate module with its own physical assumptions and validation.

Organize generation around five independently selectable components: **world mechanism, graph structure, observation policy, learning task, and evaluation regime**. A world produces a history; a task decides what the learner sees and what it must predict. Changing the learning task should not require rewriting the world simulator.

| Layer | Shared responsibility | Example for this paper |
| --- | --- | --- |
| Factory core | Versioned recipes, capability checks, stable identifiers, seeds, bounded execution, checkpoints, provenance, and artifact integrity. | Execute a reproducible recipe and resume its stateful generation. |
| World and structure modules | Define entities, interactions, state transitions, and domain-specific validity rules. | Orders, items, resources, branching, batching, and queues. Other modules can represent software, transport, collaboration, or RF. |
| Observation modules | Specify which facts are recorded, when they become available, and how recording failures occur. | Missing participant links, identifier changes, and delayed records. |
| Task modules | Define permitted inputs, queries, targets, candidate sets, and scoring. | Graph repair for the paper; future participation prediction as a second task using the same histories. |
| Export and evaluation modules | Package datasets, loaders, splits, dataset cards, and comparable baseline protocols. | A typed event graph and an OCEL adapter with explicit conversion limits. |

The common data contract should represent entities, timestamped events, typed relationships, observation availability, task queries, and provenance. Event occurrence and observation availability must remain distinct. Support event streams, snapshots, heterogeneous graphs, and multi-participant events through declared adapters; record any information lost in conversion. A graph representation is not itself a domain simulator.

Future task modules can cover link and event prediction, node-state forecasting, anomaly detection, repair, temporal question answering, graph generation, and prompt-based adaptation. Intervention and control tasks require world modules that explicitly support those operations. Capability checks must reject incompatible combinations. “Extensible to different temporal graph tasks” is the architectural objective; “all tasks are already supported” would be an unsupported claim.

For example, one generated logistics history can produce a next-interaction prediction dataset, a damaged-record repair dataset, and questions about an item's journey. These tasks use different observation cutoffs and labels but share the same execution. Genuine operational exceptions must remain distinct from recording errors; neither is automatically an anomaly label without a task definition.

The first implementation must provide the following reusable contracts:

| Component | Required behavior |
| --- | --- |
| Recipe and module registry | Declare world, structure, observation, task, export choices, seeds, and size limits; validate module capabilities. |
| Execution interface | Let modules maintain their own state and emit common typed events. Workflow queues and business rules belong in the workflow module. |
| Observation interface | Produce learner-visible records separately from reference execution and preserve availability times. |
| Checkpoint | Save module state, pending events, and random-generator state; saving only the last event index is insufficient. |
| Task interface | Produce allowed inputs, queries, targets, and scoring rules. The repair module supplies graph-edit targets, identity equivalences, and ambiguity labels. |
| Split manager | Group histories and observation variants; hold out mechanism ancestry and compositions. |
| Validator | Check shared integrity and label separation, then invoke domain and task checks for replay consistency and duplicate-answer leakage. |
| Dataset package | Include recipes, versions, provenance, labels, splits, loaders, evaluation configuration, and a dataset card stating simulation assumptions and source terms. |

Acceptance for this increment requires an integrated workflow module and two task builders over the same histories: repair and future participation prediction. The prediction task must use prefix-only observations and keep future facts in evaluator-only targets; the repair task may use its declared offline window. Use a small independent, non-workflow event module to check that the shared interfaces do not require purchasing concepts. These are architectural checks, not evidence of learning success across all domains or a second full paper evaluation.

Researchers should ultimately specify a domain module, scenario, observation policy, task, and budget, then receive a reproducible scientific dataset package. Learned generators and language models can be optional modules; they are not mandatory dependencies for every dataset. Automation can execute and validate a declared experiment, but domain realism and usefulness still require evidence. The first paper evaluates the workflow repair application; later modules expand the factory without replacing its core.

## Staged plan and decisions

The durations below are planning estimates for focused work, not completion promises.

| Stage | Work and deliverable | Continue or stop condition |
| --- | --- | --- |
| 1. Freeze the task, about 3–5 working days | Audit complete schemas, reconcile latest competitor versions, define permitted evidence and three corruption families, and create a small hand-inspected example set. | Stop or change the claim if the same setting is already convincingly solved. Stop if usable reference labels disappear after leakage removal. |
| 2. Establish the baseline, about 1 week | Implement exact joins, strong temporal matching, and constraint-based matching on logistics and the BPI 2012 development partition. Measure ambiguity and candidate recall. | If cheap methods saturate the meaningful task, do not build an elaborate model. If candidate recall is poor, repair retrieval before training. |
| 3. Build the first factory increment, about 1–2 weeks | Implement shared event, module, observation, and task interfaces; add the six workflow mechanisms and export 2,000 audited episodes. Check resume and split isolation. Build repair and prefix-only prediction datasets from the same histories, and check the interfaces with a small independent event module. | Continue only if recipes create distinct behaviors, repair labels replay to the reference history, and changing tasks does not require changing the workflow engine. Report this integration separately from model performance. |
| 4. Test the learning hypothesis, about 1–2 weeks | Train the compact edit model and matched sequence/static alternatives; compare generated data with simpler training sources. | Continue only if gains survive held-out mechanisms and a reserved BPI 2012 development partition. Keep the BPI 2017 and 2019 tests sealed. If only target-specific training works later, revise the transfer claim explicitly. |
| 5. Freeze and evaluate, about 1–2 weeks | Run both real application tests, independent simulator tests, three seeds, exception preservation, and the essential ablations. | A positive paper needs useful improvements over the strongest applicable baseline, with preserved correct facts and credible transfer. Simulated-only improvement is insufficient for the proposed application claim. |
| 6. Write and package, about 1 week | Produce the paper, method code, dataset recipes, source attribution, executable evaluation, and failure examples. | Write claims around the actual findings. Do not retain the original contribution statement if the experiments refute it. |

Provisional development targets are a five-percentage-point improvement in traceability accuracy over the strongest matched baseline on both real process families, at least 99% preservation of originally correct facts, and no material deterioration in exception attribution. These are practical go/no-go targets, not statistical significance thresholds or publication guarantees. Final coverage and edit-precision requirements must be fixed before opening the final test results.

If these targets fail, investigate the cause once in a bounded revision: missing candidate evidence, inadequate generated mechanism diversity, or a representation that adds no value. Report a negative finding if appropriate. Do not keep changing the test problem until a favorable result appears.

## The paper we would be trying to earn

A strong result would support three connected contributions:

1. A practical benchmark for joint identity and participation restoration, with explicit evidence boundaries and application queries.
2. A reusable factory increment with separate world, observation, and task interfaces, plus a workflow module supplying complete histories, damaged observations, and verifiable repair labels while retaining genuine exceptions. A second task demonstrates interface reuse; broad empirical validation remains future work.
3. A compact generative repair model that transfers to different real process families and improves traceability over strong matching, graph, sequence, and LLM alternatives.

The decisive result is useful restoration on unfamiliar real-source workflows. The generator, prompts, and model architecture are means of achieving that result. This makes the project a concrete application paper and a substantive step toward scientific temporal dataset generation.
