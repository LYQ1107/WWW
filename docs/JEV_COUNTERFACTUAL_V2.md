# Counterfactual v2 protocol

## Engine identity

`cached_perception_mutable_association_v2` freezes detector/ReID perception
per frame and view, while copying and mutating the tracker state per branch.
The cache contains:

- predicted boxes;
- detection scores;
- ReID features;
- image size;
- proposal metadata and source metadata;
- per-record checksum and a cache version.

The cache writer is enabled only with `JEV_PERCEPTION_CACHE_PATH`; the default
runtime path is unchanged.

## Branch protocol

For a pre-action state `S_t`, each legal action receives a deep copy. The
future branch uses the same cached perception records and reruns association
against the branch-local track prototypes/state. Memory and stale-bank state
are branch-local as well. The formal association implementation must inject
the GMT association transformer; the repository currently also provides a
deterministic cosine backend for protocol tests, explicitly marked
`formal_gmt_association_adapter: false`. The `GMTAssociationTransformerAdapter`
now reconstructs the GMT association window and calls the repository
transformer; the current full-sequence shards are still running and require
fixed-model replay validation before their output is accepted as formal
evidence. All formal v2 artifacts must bind to canonical Stage2
`model_20000.pth` (SHA256
`cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`).

## Raw utility fields

Each branch stores raw values before scalarization:

`future_correct_identity_duration`, `future_identity_switches`,
`future_fragmentation`, `future_collisions`, `memory_contamination`,
`contamination_duration`, `recovery_latency`, `false_reactivation`, and
`new_id_fragmentation`.

The scalar utility is a documented analysis choice, not the only reported
outcome. Uninformative windows receive zero sample weight rather than an
implicit `SKIP_MEMORY` label.

Each shard manifest also locks the checkpoint, config, source commit, engine
version, state schema, utility definition, association backend, annotation/cache
and trace digests, horizon, and authority flags. The merge tool refuses to mix
different provenance, even when the checkpoint digest is the same. A low-memory
implementation must first pass the bounded builder equivalence check (record
count/state/legal actions/best action exact, utility within the declared
tolerance).

The formal policy suite uses one `Hmax=32` rollout. Each record preserves raw
cumulative outcomes for `H=1/8/16/32`; smaller horizon datasets are derived
from that evidence and do not rerun the association simulator.

## Limitations before formal use

The v1 trace and labels cannot be relabeled as v2 merely by renaming a file.
Formal v2 evidence requires a cache generated from the fixed proxy/final GMT
checkpoint and a verified association-transformer adapter with cache/state
replay equivalence checks. TEST generation is fail-closed until the canonical
selection lock exists; an already-running pre-lock TEST diagnostic is marked
`QUARANTINED_PRELOCK_TEST_DIAGNOSTIC`, gets a `DO_NOT_USE_FOR_SELECTION`
sentinel, and cannot affect policy, horizon selection, or the final report.
Official TEST is regenerated after the lock in a separate `test_official`
namespace.

The OFF validation is deliberately split: 4A validates the trace contract;
4B reruns the frozen cache through the formal GMT simulator and compares the
proposal, assignment, IDs, memory/stale-bank transitions, and final trajectory.
4B must pass before counterfactual labels become authoritative.

Every official TEST shard and merged manifest must also carry
`official_test_generation_authorized: true` and the `sha256:` digest of the
canonical `FINAL_SELECTION_LOCK`. This prevents a pre-lock diagnostic shard
from being reused after the pipeline restarts.
