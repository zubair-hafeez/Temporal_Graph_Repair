# Literature refresh during implementation

Checked 15 October 2026. This augments the prior novelty audit; no completeness claim is made.

## SANAGRAPH final publication

The publisher records *Graph-Based Event Log Repair*, CAiSE 2026, pp. 423–440, first online 9 June 2026, DOI [10.1007/978-3-032-28117-3_24](https://link.springer.com/chapter/10.1007/978-3-032-28117-3_24). Its abstract retains the task of restoring missing attributes within supplied traces, and the publisher links the same SANAGRAPH repository. Full chapter text is subscription-only in the retrieved page. Author repository access returned HTTP 403, so full version reconciliation remains incomplete. The accessible abstract and links support the same task boundary as the inspected preprint; they do not prove all experimental details are unchanged.

## Additional generative competitor

The final SANAGRAPH reference list identifies *DDTR: Diffusion Denoising Trace Recovery* (ICPM 2025). The [primary paper](https://arxiv.org/abs/2510.22553) and [author code](https://github.com/maxim-mat/DDTR) are available. DDTR reconstructs discrete activities from probability-valued activity sequences, optionally using process-model guidance. Its input retains trace grouping. This is relevant generative denoising prior work, but it is not an unchanged baseline for event-to-object identity reconstruction. If adapted, describe the changed input/output and shared information explicitly. Never claim the first generative event-log repair method.

The publisher also cites *A Reinforcement Learning Framework for Event Log Anomaly Detection and Repair*, Ko et al., ICPM 2025. The linked [author repository record](https://eprints.qut.edu.au/259558/) could not be retrieved. Locate accessible primary full text before freezing the anomaly/repair comparison. The title alone does not establish its assumptions or comparability.

## Application need and benchmark overlap

The [object-centric imperfection patterns study](https://doi.org/10.1016/j.is.2026.102766) includes object clones, missing event participation, and incorrect participation, with real-source and injected-error evaluation. Its concrete detection implementations and data should be inspected as an additional baseline source. The proposed contribution must exceed a new naming scheme for established imperfections.

Current judgment: the joint identity/participation restoration hypothesis remains worth testing. Generic graph repair, cross-log evaluation, multiple error types, and generative denoising are already established separately and cannot be novelty claims.
