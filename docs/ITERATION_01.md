# Iteration 1 — observation-only baseline gate

The initial missing-participation diagnostic is implemented and has run on development data. It does not yet evaluate the proposed joint repair task or a synthetic-trained generative model.

## Protocol

Graphs expose event activity/lifecycle, occurrence time, entity type, and observed participation. IDs are shuffled opaque handles. Raw attributes, source trace ordering, object-object links, corruption masks, and reference case containers are excluded. BPI 2012 core cases are partitioned before anonymization; each partition is indexed independently. Logistics remains exploratory simulated development data.

Independent and per-object contiguous deletion remove approximately 20% of eligible participant links. The probe includes queries with no surviving co-participant. Events with several missing objects of the same type form a set-valued query. Recall at the true missing cardinality receives fractional tie credit. Missing location, type, and cardinality are supplied; this is not autonomous fault diagnosis.

Each run samples 1,000 query groups. The 128-candidate retrieval pool combines nearby occurrences, same-activity occurrences, co-participant context, normalized co-occurrence, and degree. Nine fixed scoring rules are compared. A histogram gradient-boosted matcher learns from 13 structural, temporal, and activity features on the BPI 2012 development-fit partition only. It is a strong learned baseline, not our proposed method.

## Findings from seed 0

| Development setting | Best fixed scorer, recall | Learned matcher, recall | Candidate recall |
| --- | ---: | ---: | ---: |
| BPI 2012 tune, independent gaps | 84.5% | 90.9% | 99.1% |
| BPI 2012 tune, contiguous gaps | 49.1% | 56.0% | 89.8% |
| Logistics, independent gaps | 32.9% | 50.6% | 87.6% |
| Logistics, contiguous gaps | 32.4% | 48.7% | 96.2% |

Exact results and source/code hashes: `reports/baseline_probe_v1.json` and `reports/matching_v1.json`. These are exploratory single-seed development numbers. They are not a comparison against published repair methods, a neural improvement, or final paper evidence.

Two findings determine the next experiment. First, independent missing-link restoration is already fairly easy in BPI 2012, making it a weak sole contribution. Second, retrieval is a bottleneck: on the larger development-fit partition, only 71.4% of contiguous-gap targets enter the pool. A 512-candidate diagnostic is running before any generative model is designed.

Logistics difficulty differs sharply by type. Handling units have little observed history, while containers offer more context. Some source object-object links and timed attributes could provide additional evidence; they require an explicit matched-input comparison rather than being silently omitted from the final strongest baseline. Their availability-time semantics must be documented. Missing all observations of an object cannot be made identifiable by a neural model.

The tune partition contains fewer competing cases than the fit partition. Their absolute scores are not directly comparable measures of generalization. Final experiments must report graph size and candidate population, and include size-matched development checks.

## Checks and next gate

Five unit checks passed: hidden links absent from the index, source-ID removal, deterministic serialization, multi-target/tie scoring, and immutable reference histories. These checks do not certify all absence of leakage; richer evidence and fragmentation need additional tests.

Next: improve and measure retrieval, add explicit-evidence matching and coherent assignment controls, then evaluate mixed missing/wrong relationships and identity fragmentation. Preserve clean facts and genuine exceptions as separate outcomes. Proceed to the generator only if those checks leave a recoverable application gap.
