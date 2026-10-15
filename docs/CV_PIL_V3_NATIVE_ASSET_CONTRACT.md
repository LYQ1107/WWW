# CV-PIL V3 native development consumer

2026-10-15. Consumer branch: `research/cv-pil-fast-screen-20261015`.
Interface: **`cvpil-dev12-native-assets/v1`**. This supersedes the proposed V2
header for this new DEV12 experiment, without altering any old experiment.

## Semantic mapping, without prescribed filenames

`geo_tools/cv_pil_dev12_assets.py` provides a canonical read-only mapping. A
producer may use other filenames or equivalent serialization. A different
layout needs an explicit, tested mapping of the same semantics; it must not
be rejected merely for naming, or admitted merely by changing its header.
The reference mapping uses five distinct SHA-bound artifact roles:

| Role | Required semantics |
|---|---|
| `inputs` | Observations and complete association calls; no supervision |
| `features` | Full frozen native 1152-D tensors indexed by physical observation |
| `actions` | Pre/post state hashes, original assignment and thresholds, accepted/Bank/birth/final IDs, Gallery/Bank update hashes |
| `native_context` | Separately sealed original-trace call anchors: all row keys, all candidate Global IDs, original matrix digest, physical observation metadata digest, action digest |
| `offline_supervision` | Physically separate person/alias/impurity/UNKNOWN/primary certification labels |

The manifest is externally SHA-pinned, `status=FROZEN_COMPLETE`. The constructor
also requires independently obtained `expected_provenance`, `expected_scope`
and an authorized read-only root. Never build these expectations by copying
the unverified candidate release's own declarations. Source bindings include
the original full Stage2 and config SHA, source commit, video, physical camera
map and GT SHA. A digest certifies immutable bytes, not the truth of a claim:
the caller must establish that `native_context` is derived from the original
sealed native trace, not from the exported shortlist.

The frozen scope specifies DEV12, all1200 synchronized groups and all actual
primary queries; selection is label-free and frozen before export. Query count
and the digest of the ordered physical query keys must match the independent
scope and the actual calls. A preregistration SHA must be provided. Previously
reported14557 queries are a consistency reference, not permission to silently
truncate a different source to that count. Empty-query groups need producer
scope evidence; query rows alone do not certify their existence. TRAIN12 is
exposed development, never blind validation.

## Complete calls and causal features

An observation contains `[video,image_id,original_detection_order]`, scene frame,
physical camera1/2, a native causal index, and feature reference. Each call
contains original ordered query keys, **all** legal candidate IDs, complete
native track scores and solver matrix, reference observation keys and owners,
reference block sizes, logits, and per-candidate history sources, lifetime
observed length and current reference support count. Native matrices must
match original trace anchors, including rows belonging to other detections.
No Top8 reconstruction, GT-created candidate or missing-feature interpolation
is accepted. Empty candidate calls are valid and retain original birth semantics.

Native scores are independently reconstructed: per historical reference block,
append a zero unmatched logit, softmax, remove unmatched, then sum probabilities
by native Global ID. CPU reconstruction permits atol3e-5/rtol4e-6 for GPU/CPU
softmax and accumulation differences; the original matrix itself remains exact.
Original matrix
JSON bytes are additionally bound by digest. Candidate history ownership and
support counts must match the full reference list. Original Hungarian pairs
must be one-to-one, complete and attain the full matrix optimum, with original
strict `score > threshold` acceptance. This audits a recorded primary solve;
it does not integrate a new controller or replay Bank under changed decisions.

Every native reference must precede the query's native causal index and cannot
come from a later scene frame. Same-frame references from an earlier camera
decision remain in the original GTA/solver context. For the old frozen heads,
the model projection uses only strictly earlier scene frames (positive lag),
all such histories and **all legal candidates**. Same-frame references are
explicitly excluded from the head, never removed from the native solver.
This is a frozen, label-free preprocessing rule, shared by C1/C2/C3/C4.
Unlike V2's selected Top8/latest4 payload, V3 evaluates complete histories;
the distribution change must be reported and V2 scores must not be pooled.

## Supervision and inference isolation

`actor()` produces the existing exact GT-free model input schema and cloned
full-dimensional tensors. It never opens `offline_supervision`.
`supervision()` is an explicitly separate offline operation with a fresh SHA
check. UNKNOWN/impure candidates have no negative label; multiple pure native
IDs for one person are all positives. Label-derived candidate creation is an
error. Known wrong/correct/missed primary labels are checked against original
accepted IDs and pure candidate ownership. No numeric predicted-ID=GT-ID test
is used. Final identity correctness and changed Bank/birth outcomes require
their own certification, not a rank or a new numeric ID.

## OFF and validation limits

`off_passthrough` calls the supplied original function with unchanged arguments
and returns its exact object. Adversarial tests cover this call boundary,
prediction bytes and state identity. **They are synthetic unit tests, not a
certificate of full original GMT video parity.** No new module has been
connected to the full tracker. Real OFF predictions, Gallery/Bank state and
raw/stock outputs must be verified on the actual same full range at gated
integration. This obligation is explicitly pending, not recorded as PASS.

The adapter is currently tested on synthetic contracts only. New producer
DEV12 assets have not been admitted. No model was retrained and no new native
tracking result is claimed.

Run CPU tests:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/liuyeqiang/anaconda3/envs/GMT/bin/python -m unittest discover -s geo_tools -p 'test_cv_pil_*.py' -v
```

## Actual producer shard mapping and frozen-head runner

`cv_pil_dev12_shards.py` now maps the actual producer layout, with role paths
supplied explicitly rather than mandatory filenames. It imports no producer
research code. Full consumption requires both independently pinned manifest
and final artifact-index SHA, the already-frozen scope SHA and original source
bindings. Every bound file, complete score matrix, original threshold action,
physical query/commit link and commit-state hash chain is checked.

Native Bank can mutate a descriptor while retaining its original physical
source tag. The mapper preserves the actual SHA-bound descriptor version;
it never substitutes the raw ROI vector or creates an average. The original
Bank descriptor is additionally checked against its recorded ordered native
support/denominator. Offline pure/impure/alias annotations remain separate.

The independently frozen real40 prefix was accepted **for engineering only**:
626 queries/79 calls, every full actor tensorized, no GT read, no head evaluated.
This path explicitly refuses `supervision()`. Raw matrix Hungarian actions are
exact; maximum CPU aggregation difference was1.52587890625e-5, disclosed rather
than treated as a bitwise GPU softmax match.

`run_cv_pil_v3_old_heads.py` is prepared for P2 after full sealed admission.
It scores all rows/candidates before opening labels, loads only old step300
weights, and compares full joint primary assignments with the same recorded
thresholds and starting state. This is one-step causal primary replay, not
mutated-state online tracking. Changed Bank/birth and final committed errors
remain unmeasured; original post-actions cannot be reused as counterfactual
post-state. HOTA and full-image FPS stay null. Grouping includes person,
person/camera and fixed50-frame intervals, never independent candidate edges.
