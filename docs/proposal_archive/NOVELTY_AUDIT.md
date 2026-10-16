# Temporal graph research selection and novelty audit

Assessment date: 15 October 2026. Recommended candidate: **learning to restore operational temporal graphs from generated workflows**. See [the proposal](PROPOSAL.md) for the task, method, datasets, and decision gates.

This is a targeted research assessment. Publication prospects depend on the proposed experiments. The search does not establish that no related work exists. Source claims, local observations, and proposed contributions are distinguished below.

## Selection criteria

The new project must address a recognizable application problem, make time and relationships useful to solving it, fit a small research team with two GPUs, and provide a substantive direction beyond one metric or one missing field. Generation should supply useful training examples or produce a useful output. Prompting should encode a real task requirement. RF is excluded.

The existing forecasting paper remains a separate project. Its models and evaluation labels are not evidence for this proposal.

## Candidate screening

The rows are distinct candidate families or substantial revisions, rather than a claimed count of independent experimental iterations. Review depth differs: direct competitors received method or code inspection; other branches were screened using primary abstracts, project descriptions, and the earlier local audit.

| Candidate | Finding | Decision for this project |
| --- | --- | --- |
| Generic few-shot temporal graph prompting | DyGPrompt and FITTER in the local collection already offer closely related adaptation ideas. [STGP](https://github.com/hjf1997/STGP) also covers spatiotemporal adaptation across domains and tasks. | A new prompt module alone does not provide the application gap we need. |
| A general prompt-driven temporal dataset factory | [SAGA](https://arxiv.org/abs/2607.17288) already combines temporal graph generation, domain semantics, replay, and anomaly labels. | A factory is infrastructure. Its research value must be demonstrated on a specific unsolved task. |
| Synthetic pretraining for relational or temporal models | [PluRel](https://arxiv.org/abs/2602.04029) already demonstrates synthetic relational pretraining and transfer to real databases. | Do not claim synthetic pretraining itself as the novelty. |
| More realistic operational workload generation | The [previous audit](../first_paper_20261014/NOVELTY_AUDIT.md) documents TraceGen, AgentSimulator, RIMS, TraceLLM, and Mimesys. | Too close to the previous paper and its statistical evaluation emphasis. |
| Prompted fault diagnosis across microservice deployments | [MetaRCA](https://arxiv.org/abs/2603.02032), accepted to FSE 2026, explicitly addresses cross-topology generalization using reusable causal knowledge. | Generic graph plus LLM diagnosis is insufficient. A new execution task would need a separate, expensive systems evaluation. |
| Graph-guided repair of failing software agents | [AgentTether](https://arxiv.org/abs/2607.06273) uses temporal and artifact dependencies for failure localization and guarded intervention. [ORCA](https://arxiv.org/abs/2608.17018) targets microservice incident repair. | Strong direct competition and substantial environment engineering. Not selected. |
| Prompt-generated traffic or navigation scenarios | [STGT-Gen](https://doi.org/10.4271/2025-01-7316) generates traffic scenarios. [DynamicTHOR](https://www.nature.com/articles/s41597-026-07201-7) supplies human-driven household dynamics and navigation tasks. | Generic scenario generation is already established. Additional simulation/perception work would dominate a first paper. |
| Supply-chain temporal graph learning | [SC-TGN and SupplySim](https://cs.stanford.edu/people/jure/pubs/supplychain-gnn-aaai25.pdf) already model production and transactions. | A promising application family, but another forecasting model would not meet the requested change of direction. |
| Generative repair of missing event attributes | [SANAGRAPH](https://arxiv.org/abs/2508.05145) repairs attributes with a heterogeneous GNN. | Reject this generic formulation. |
| LLM repair of several log errors together | Schmid, Zetzsche, and Röglinger's *One to Rule Them All* already combines multiple imperfections and includes an unseen-log evaluation in the [2025 dissertation](https://epub.uni-bayreuth.de/id/eprint/8484/1/Dissertation_Sebastian_Schmid.pdf). | Neither multi-error repair nor cross-log transfer alone is new. |
| Missing object links as an isolated ranking task | The [OCEL Relation Gaps implementation](https://promenade.run/plugins/relation-gap/) reconstructs missing event-object links using object co-occurrence and a GraphSAGE comparison. | Reject “first GNN for missing object links.” Use this as a baseline. |
| Joint restoration of object identities and event participation in unfamiliar workflows | The nearest inspected methods address important components, but do not demonstrate the proposed combination of generated workflow training, coupled identity/relation restoration, and preservation of exception attribution on held-out operational processes. | Select as a testable research hypothesis, with the novelty boundary below. |

## Closest competitors and claim boundaries

### SANAGRAPH

The inspected [paper](https://arxiv.org/html/2508.05145v1) represents attributes within a trace as a heterogeneous graph and reconstructs missing values. The inspected [repository](https://github.com/sebdisdv/SANAGRAPH), commit `c26365f014b0e0ff81574df682b359871b1301f8`, extracts traces using `CaseID` before graph encoding. The local code excerpt is in `evidence/sanagraph_encoding_reference.txt`.

Our proposed input lacks reliable assignment of events to several interacting objects. It cannot assume correct case grouping before repair. Nevertheless, SANAGRAPH is a strong attribute-repair comparator, and a modified version with the same candidate information must be evaluated. The inspected arXiv version is dated August 2025; the work also has a 2026 CAiSE publication record. The final published method must be reconciled before experiments are frozen.

### HERE and unified LLM repair

[HERE](https://link.springer.com/article/10.1007/s12599-025-00935-5) combines Transformer prediction, expert rules, and human input for missing case identifiers. Its stated assumptions include complete clean training data and an isolated case-identifier imperfection.

The subsequent [unified LLM repair work](https://epub.uni-bayreuth.de/id/eprint/8484/1/Dissertation_Sebastian_Schmid.pdf) is broader: it fine-tunes LLaMA-3.1-8B on corrupted real logs, processes batches of up to 150 events, and handles case ID, activity, and timestamp errors. It also reports an unseen-log test. Our proposal must therefore demonstrate useful restoration of **multi-object structure and identity**, not merely add a temporal encoder to CSV repair.

### Object link repair and general data cleaning

The [relation-gap plugin](https://promenade.run/plugins/relation-gap/) is implementation evidence, not evidence of a peer-reviewed benchmark. Its release notes disclose a correction to masking during GraphSAGE training. Reproducing its corrected masking behavior matters more than reproducing an earlier headline score. The underlying student report was not independently located during this audit.

[HoloClean](https://arxiv.org/abs/1702.00820), [Ditto](https://github.com/megagonlabs/ditto), and [graph repair for structured event logs](https://doi.org/10.1145/3571281) preclude claiming that combining constraints, learned matching, and graph edits is itself new. Their components support a strong sequential pipeline comparator.

### Synthetic data and real need

[PluRel](https://star-project.stanford.edu/plurel/) generates schemas, connectivity, and feature mechanisms. The proposed question is whether explicit execution histories, observation corruption, and verified repair targets add value for operational restoration at a matched data budget. Generic synthetic graphs with random corruption are a necessary control.

[Object-centric event-data imperfection patterns](https://doi.org/10.1016/j.is.2026.102766) establishes the practical relevance of identity and relationship problems and gives detection/remedy approaches. This supports the application need; it does not establish that our proposed learner works. [Workarounds as a Cause of Mismatches in Business Processes](https://link.springer.com/article/10.1007/s12599-025-00943-5) further motivates distinguishing actual behavior from what its records appear to show.

## Proposed novelty statement

**Hypothesis:** a model trained to generate coordinated graph edits on diverse simulated workflows can restore event participation and fragmented object identities in a previously unseen operational process, using its schema and available context, with better traceability and fewer destructive edits than independent repair stages.

The contribution would comprise an operational restoration task, a generator that separates true behavior from recording corruption, and a transferable repair model. None of those three components is sufficient alone. A successful paper must show that their combination changes application outcomes on real-source logs.

Avoid claims of the first graph repair method, first LLM log repair, first synthetic temporal generator, first cross-log transfer, or a universal temporal foundation model. Also avoid claiming recovery of uniquely unknowable histories or reliable repair of naturally occurring errors without independent truth.

## Local feasibility evidence

The [data audit](data_feasibility.json) verifies publisher checksums for downloaded logistics and Angular data. The [purchasing audit](bpi2019_feasibility.json) verifies the public BPI 2019 conversion archive and inspects a bounded sample.

| Check | Observation | Implication |
| --- | --- | --- |
| Logistics file, pinned Zenodo record 18373888 | 35,372 events, 13,882 objects, 74,272 event-object links, and 22,202 events with several objects. | Concrete multi-object development data exists. It is simulated. Counts differ from the overview website, so use the file manifest. |
| Existing BPI 2017 raw data, first 500 applications | 20,295 events and 768 explicitly recorded offers; 178 applications have more than one offer. | The object graph has meaningful structure beyond one case ID. Only explicitly observed offer links can serve as reference labels. |
| BPI 2019 conversion, first 2,000 events | Purchase order, item, resource, and vendor types; 7,470 participant links in the inspected events. | Purchasing provides a second real process family and useful traceability tasks. |
| BPI 2019 duplicate fields | Participant identities are also present in several event attributes. | Merely deleting a graph edge produces a leaky benchmark. Corruption must affect every representation of the same evidence. |
| Angular GitHub log | The earliest 2014 commit is linked to 119 branches. | These may be retrospective memberships. Exclude it from the main temporal evaluation unless branch-validity times can be established. |

A small logistics probe evaluated 2,000 known relation holes per corruption mode and seed, for three seeds. A simple temporal-neighbor score recovered about 44–45% of the queried identities, while an unnormalized co-occurrence count recovered about 2–3%. These are **exploratory heuristic results on simulated data**, with the missing relation type supplied and both earlier and later observed context allowed. They establish neither neural-model benefit nor superiority over strong static baselines. Their useful implication is that time contains recoverable information and that nearest-neighbor repair is a mandatory competitor. Corruption severities were not matched across the two modes.

## Remaining falsifiers

1. A stronger recent method already solves the complete proposed setting. Reproduce or extend that method only if an independently useful failure remains.
2. Role-aware temporal matching plus constraint optimization saturates the task. Do not add a neural model for decoration.
3. Most errors become unresolvable after duplicated identifiers are removed. Reduce the promised coverage and test review assistance, or abandon the autonomous restoration claim.
4. Generated training data fails to improve transfer to either real process family. Do not call simulated-only success application validation.
5. Apparent gains depend on access to hidden clean adjacency, exact corruption masks, source row indices, or known case grouping. Repair the protocol and rerun before interpreting results.
6. Repair improves average link accuracy but erases or misattributes genuine exceptions. The application objective has failed.

The recommendation passes the present need, prior-work, data-access, and implementation-scope checks. Learned performance and publishability remain experimental questions.
