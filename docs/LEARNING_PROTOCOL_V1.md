# First learning experiment

This protocol is recorded before the proposed model is trained. Development findings may change the design, but final target outcomes must remain sealed.

## Data

Start with 512 training executions and 96 validation executions. Eight training mechanism compositions and three disjoint validation compositions are defined in `scripts/generate_workflow_pilot.py`; reserved test compositions are defined separately and are not opened for tuning. The three named schema profiles share one simulator implementation. Treat their variation as vocabulary variation, not independent real domains.

Use a factorial schedule over mechanism composition, vocabulary profile, and flat versus multi-object observation. Randomize queue/resource settings, instance size, attribute collisions, and error rates independently. Every execution has clean and mixed-observation variants, grouped into the same split. Training rates vary over the declared grid; no target-test labels choose generator settings.

The first collection is development-scale. Passing this experiment would justify larger training, harder independently implemented observation faults, and the reserved public-source confirmation protocol. It does not authorize treating simulated validation as real-world success.

## Comparisons

1. Retain the strong source-trained participant matcher and its history-retrieval result.
2. Add a learned identity matcher with generic temporal, contextual, and descriptive evidence; evaluate it alone and after participant repair. The hand-designed identity baseline is inadequate as the sole comparator.
3. Train the same matching heads on generated observations to test the value and limits of synthetic supervision before attributing gains to a graph network.
4. Implement a compact schema-conditioned temporal graph edit policy with ADD_PARTICIPANT, REMOVE_PARTICIPANT, MERGE_ALIASES, STOP/abstention. Recompute observed neighborhoods after edits. Its action generator must not receive reference identities or the fault mask.
5. Compare coordinated edits with the same learned components applied in a fixed sequence, and compare graph/temporal representations at matched inputs and candidate budgets.

## Guardrails on interpretation

- Candidate recall is an upper bound, not model accuracy. Report it for participation and identity candidates.
- Successful alias merging must improve object-history retrieval and preserve clean entities; high recall obtained by merging many objects is a failure.
- The already inspected BPI 2012 tune data are development. Use the reserved check partition for a later bounded confirmation after development decisions, and use BPI 2017/2019 only after final freezing.
- Pure synthetic training, source-real training, mixed training, and target-support adaptation are separate regimes.
- The first neural architecture need not survive. Keep the simpler model if the graph network adds no application benefit, and change the paper's claims accordingly.
- Capture a source archive before each substantial experiment. Preserve failed runs and record exact repairs to evaluation code.
