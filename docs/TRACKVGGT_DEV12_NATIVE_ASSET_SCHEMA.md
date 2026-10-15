# TRAIN12 native development asset interface

Interface: `cvpil-dev12-native-assets/v1`. This is an engineering input release,
separate from TrackVGGT teacher fitting and the CV-PIL algorithm. The full,
continuous groups1–1200 rule is frozen before capture, independent of GT/error
locations. TRAIN12 labels were viewed previously; this is expanded development
data, not an unseen blind evaluation. Original V2/V3 membership is unchanged.

Use only a release whose manifest says `SEALED_ASSET_READY` and whose
`artifact_sha256.json` matches the producer's published SHA. That index hashes
the manifest and every other artifact; its own SHA is the external trust anchor.
Source, statistics and SHA summaries belong in Git. Native vectors/matrices stay
in the versioned server directory. Files and directories are made read-only.

## Read without GMT, videos or GT

The producer publishes the final path and SHA in the handoff report. With that
path and SHA, add this repository's `geo_tools` directory to Python's import path:

```python
from read_dev12_native_release import NativeRelease

data = NativeRelease(release_path, expected_seal_sha256=published_sha)
for key in data.queries:
    item = data.query(key)
    current = item['current_feature']  # native FP32[1152]
    histories = item['histories']    # every actual candidate, original order
    joint = item['call']             # entire real Hungarian context
    final_id = item['final_track']
```

By default no labels are loaded. Explicit `with_offline_labels=True` adds an
`offline_label` field after reading the same fixed query keys. Labels never
construct or filter the inputs. The independent validator restores all original
joint assignments, strict thresholds, complete histories and final submissions.
The sealed-reader smoke test traverses every query through this public API.

## Physical observations and native calls

|Artifact|Content|
|---|---|
|`frozen_scope.json`|Continuous clock/camera order, all image hashes, original checkpoint/config/model/capture code hashes, independent budgets|
|`capture_manifest.json`|Original completed, unlabeled capture receipt, preserved before the final manifest update|
|`native_observations/observations_*.pt`|Unique `(image_id,detection_row)` sources, frame, physical camera, original box/score, native box and lossless FP32[1152] vectors|
|`native_observations/descriptor_variants.pt`|Explicit actual descriptor versions when a native Bank operation changes a vector that shares an older physical source tag|
|`native_queries.jsonl.gz`|Every primary `(12,image_id,detection_row)` key, call ID, actual row, primary accepted ID and final ID|
|`native_calls/calls_*.pt`|Full primary and Bank calls, all rows/columns, raw logits, actual single-reference probabilities, actual aggregated FP32 solver scores, thresholds, candidate IDs, complete source order, support lengths and state summaries|
|`actual_decisions.jsonl.gz`|Actual original Hungarian pairs, threshold outcomes, Bank and final native return IDs, exactly aligned to the sealed audit|
|`native_commit_trace.jsonl.gz`|Every detection's actual commit, including initial largest-camera births, state fingerprints and a hash chain|
|`offline_labels/*.jsonl.gz`|Separate original-GT certifications, unknown/mixed histories, aliases and native action labels|
|`coverage_audit.json`|All-query completeness and individual coverage/reachability of the original26 known errors|
|`parity_audit.json`, `precheck_gate.json`|Full original-output/decision parity, real export-off/on40-state gate and independent solver replay|
|`resource_ledger.json`|Independent DEV12 outer subprocess wall times; includes loading, hashing and export|
|`consumer_audit.json`|GT-free/video-free input validation and exact separate offline-label join|
|`source_provenance.json`, `artifact_sha256.json`|Immutable capture source and complete artifact hashes|

Torch files contain ordinary dictionaries/tensors, readable with `torch.load`
after checking their published SHA. Native1152 includes original GMT visual and
spatial/camera information. Nothing is replaced by a1024-dimensional identity
embedding, normalized, rounded to FP16 or teacher-encoded.

`call.track_ids` is the original sorted Global ID column order. The query row
indexes the whole `track_scores` matrix, rather than an independently solved
Top8 row. `raw_logits` and `single_reference_probabilities` follow the exact
reference-source column order. `reference_block_sizes` preserves native history
blocks; `activation_block_sizes` records the actual primary/Bank softmax blocks.
`query_native_context_indices` gives the original query positions in the full
native context. `observed_track_lengths` records total native support before the
call; `actual_candidate_support_lengths` counts the references actually used in
that call's window. The API preserves both lengths.

The original joint CPU Hungarian input is saved at native FP32 precision and
replays exactly. CPU recomputation of GPU softmax/aggregation is checked with
disclosed numerical tolerance; it is not described as bitwise GPU equality.
Acceptance is the original strict `score > threshold`, including the original
support-length multiplication when enabled. No Top1 surrogate replaces it.

## Causal history and Bank interpretation

All history sources precede the query in actual native processing order. The
unchanged GMT can use an earlier processed camera in the same synchronized group;
this is disclosed, rather than incorrectly requiring every reference frame to be
strictly smaller. No unprocessed camera or future group is supplied. Complete
references remain in their actual native order; deduplication only shares vector
storage. Histories are not reduced to the teacher's latest-four summaries.

An averaged Bank descriptor's legacy source tag is **not** a single detection
feature or proof of GT purity. The exact descriptor version is stored separately,
with ordered averaging support and the original denominator. The consumer checks
that support's causal sources and the resulting actual vector. Original detector
vectors are never overwritten. Source keys and descriptor-version keys are
different concepts; there are no duplicate physical observation keys.

Primary acceptance, Bank solver participation/restoration, newborn creation and
the final submitted ID remain distinct. A missed primary match is not automatically
a wrong final identity. State fingerprints cover actual counts, descriptor state,
eligible Bank IDs and Bank contents after each commit. The consumer reconstructs
all per-commit support counts and verifies the state/hash chain and final IDs.
The old full audit did not contain internal state fingerprints; full-run equality
therefore concerns every recorded actual solver/threshold/Bank/final-ID result
and prediction. Internal export-off/on fingerprints are separately exact on40
real groups.

## Offline certification and allowed interpretation

All regular query certifications are reused by exact physical key from the
sealed full audit. Only initial-camera births, absent from primary-query records,
need an original one-frame GT match after full inference. This does not rerun
the24-video census or the12-scene error mining. Raw GT conflicts/unknowns remain
masked. No ideal GT Gallery, repaired identity, future observation or GT query
vector is provided.

Tags include `CORRECT_LINK`, `WRONG_LINK`, `MISSED_KNOWN_POSITIVE`,
`CORRECT_REJECT`, `CANDIDATE_MISSING`, `AMBIGUOUS_REFERENCE`, `IMPURE_HISTORY`,
`UNKNOWN_QUERY`, `UNKNOWN_CANDIDATE_COVERAGE`, `ALIASED_IDENTITY`,
`BANK_RESTORED` and `NEWBORN`. Tags can overlap. Pure-track GT maps and ambiguous
tracks are explicit; unknown/impure references are not certified negatives.

Reachability means a certified pure positive is present in the actual legal
candidate set. Top8 reachability and current-score threshold passage are reported
separately. None proves that a learned method repairs a joint assignment, changes
future Bank state, improves HOTA or has independent VGGT/predictive benefit.

`DEV12_NATIVE_ASSET_HANDOFF_COMPLETE` is an engineering status. It does not imply
`VGGT_SCIENTIFIC_SUCCESS` or `CV_PIL_SCIENTIFIC_SUCCESS`.
