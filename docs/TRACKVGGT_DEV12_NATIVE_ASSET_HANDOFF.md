# TRAIN12 native asset handoff: complete

Engineering status: **DEV12_NATIVE_ASSET_HANDOFF_COMPLETE**.
Interface: `cvpil-dev12-native-assets/v1`; sealed manifest status:
`SEALED_ASSET_READY`. This release supplies inputs to the CV-PIL consumer; it
does not train CV-PIL, modify another branch's weights, or establish VGGT/CV-PIL
scientific success.

|Acceptance question|Executed result|
|---|---|
|A. Was delivery successful?|Yes. Native inputs, offline labels, all hashes, parity and actual read-only consumer checks pass; server artifacts are read-only.|
|B. Development coverage?|All1200 groups/2400 images,14557 actual primary queries,14564 original observations and commits. Initial7 largest-camera births remain separate from association queries.|
|C. Complete inputs for the26 errors?|26/26, including full legal candidates and all actual native history vectors. No source-key or feature/matrix gap remains.|
|D. Reachability?|23/26 have a certified pure positive in full legal candidates;21/26 in Top8;2 full positives outside Top8;3 have no certified pure positive. Current native scores pass a positive's threshold in6/26. These are distinct diagnostics, not method corrections.|
|E. Resource use?|0.340162307 GPU-wall-hours on physicalGPU5, including export-off/on40 prechecks and full process startup/loading/hash/export. Final payload281762969 bytes; allocated file blocks281862144 bytes. Limits:4 GPU-wall-hours and4GiB, independent of teacher extraction/fitting.|
|F. What should CV-PIL read?|The versioned local directory and published seal SHA below. No large feature/matrix upload is required.|
|G. Original VGGT research?|V2/V3 indices, seeds, GMT core and old audits unchanged. Original AG/ROI/DINO/native-MLP queues execute separately from immutablee28d799e; expanded DEV12 assets are not used for their fitting or model selection. QD/CVTIC/conditional IRGD remain independent pending research, not completed by this handoff.|

## Read-only location and exact SHA

Directory:

`/data1/liuyeqiang/WWW_trackvggt_runtime/cv_pil_dev12_native_release_v1/`

External trust anchor, SHA256 of `artifact_sha256.json`:

`ef7f65d0bccb1e665f7125391f4a3c64a1449b1fe4d204c735371522386421d4`

SHA256 of `manifest.json`:

`a4d2014edcc920f45753e0f76c378bf08f25c22758e3d8770572dc36632c2173`

Frozen pre-capture selection manifest SHA:

`45824abd0f27f105e4af28f8fb0ce10790ac5adc9eee1a2c579689786f5bcaf9`

The seal index hashes the manifest and every other raw artifact. Its own SHA is
published here and in `results/trackvggt_dev12_native_release_v1.json`.
File permissions are0444; directories0555. The producer retained the original
unlabeled capture manifest before adding separate offline supervision.

## Real equality and consumer evidence

Capture source: immutablef97cf27f0f4299269c3bd1dd885a92e90639f51c.
Continuous original GMT starts at group1 and ends at group1200 with seed20261014.
Original411-key checkpoint remains SHA
`cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`.
No VGGT/DINO is invoked during native export; all1152 dimensions are native
FP32 with lossless storage. Repeated observations are shared by physical key.
Sixteen actual descriptor versions preserve Bank means separately from detector
vectors, with their ordered real support.

The export-off and export-on40-group runs match original predictions exactly,
and all40 original Gallery/Bank state fingerprints match each other. Fourteen
CPU contracts pass. Full1200 predictions have identical file bytes to the old
sealed TRAIN12 audit. All2477 actual decision records match exactly:2399 primary
calls and78 Bank calls. The independent consumer restores every full joint
Hungarian assignment, strict threshold and final submission; it reconstructs all
per-commit support counts and checks the state/hash chain.

The old full audit did not save internal state fingerprints, so the full-run
claim concerns actual recorded solver/threshold/Bank/final-ID results and exact
predictions. The40-group internal-state gate is additional, separately scoped
evidence. CPU softmax and aggregation reconstruction use disclosed tolerance;
maximum absolute differences are2.384185791015625e-7 and3.814697265625e-5.
The saved actual FP32 CPU solver input replays exactly; no row-Top1 substitute is
used for joint assignment.

After sealing, the public `NativeRelease` API actually traversed all14557
queries in39.361s, materializing166439 candidate histories and6972936 history
vector references. It checked every artifact SHA and final commit, loaded no GT
labels, read no images and made zero GMT inference calls. This is an offline
consumer test, not deployment FPS. Its external receipt is
`results/trackvggt_dev12_sealed_reader_v1.json`; it does not mutate the sealed
release.

Example from this repository with its existing Python environment:

```python
from read_dev12_native_release import NativeRelease

data = NativeRelease(
    '/data1/liuyeqiang/WWW_trackvggt_runtime/cv_pil_dev12_native_release_v1',
    'ef7f65d0bccb1e665f7125391f4a3c64a1449b1fe4d204c735371522386421d4',
)
for key in data.queries:
    item = data.query(key)
    # item['current_feature']: FP32[1152]
    # item['histories']: complete actual candidate support, original order
    # item['call']: full matrix, Hungarian, thresholds, Bank/context metadata
    # item['commit']: actual final ID and state fingerprint
```

Default construction does not load labels. An explicit
`with_offline_labels=True` adds the separately joined certification; it does not
change the fixed input membership. Detailed tensor/source semantics are in
`docs/TRACKVGGT_DEV12_NATIVE_ASSET_SCHEMA.md`.

## Original26 errors and remaining scientific limits

The original6 accepted wrong links and20 missed known positives all have complete
inputs. The6 accepted wrong links retain their primary ID; all20 rejected
positive queries create newborn IDs. None of these26 is Bank-restored. The
whole scene does contain one accepted Bank restoration, preserved in its real
trace. A newborn numeric ID is not by itself proof of final GT identity error.

Errors involve7 GT people:1,3,4,5,7,8,12. Person5 contributes14/26, so these are
correlated events rather than26 independent people. Camera2 contributes21 and
camera1 contributes5. Events occupy10 of the12 fixed100-group intervals; exact
frames, people, cameras, candidates, Top8 and individual gap explanations remain
in `coverage_audit.json` and its compact published summary. Three events have no
certified pure positive in the unchanged candidate set; they are retained, not
silently dropped or supplied with an ideal GT Gallery.

All14557 query certifications are reused by exact physical key from the sealed
audit. Only the7 initial non-query births require original one-frame GT matching,
performed after complete inference and native/consumer parity. Unknown, impure
and aliased histories remain explicit offline masks. No24-video census or
12-scene error mining is repeated.

TRAIN12 labels were already viewed. The selection rule is GT-independent and
continuous, but this is expanded development data, **not an unseen blind test**.
The release does not prove an algorithm corrects any of the26 events, improves
HOTA, or has unique predictive/VGGT benefit. Original VGGT fitting uses its
unchanged V2 1000/300 and prespecified queues. Initial full AG/ROI results are
negative and are preserved in
`results/trackvggt_teacher_progress_at_dev12_handoff_v1.json`; remaining
DINO/summary settings continue. No student distillation or scientific success
is authorized by the data-delivery result.
