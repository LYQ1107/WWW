# WorldMOT V3 R2 scientific diagnosis

This is an in-progress TRAIN-only audit of the already running R2 experiment,
authorized by attachment `294ae6fe-469b-43f7-9d7b-3e0a3512409e`. It does not
change training, thresholds, selected budgets, frozen assignments or TEST inputs.
The model/input source fingerprint remains
`6bcdcdb1a6b3683c4e406b0791416a77e5e9e64f165fbad4a61667698b64645f`.
No R3 training or additional seed is started.

## Measured execution state

Full no-geometry completed four selection epochs, selected two final epochs,
then actually trained all 24 scenes twice. Each final epoch processed 28,754
paired times and 539,738 clean matched identity observations. Its final checkpoint
SHA is `b08badfc634c9652a34366951c93207fa321deb7d1a2b240a76798f11607053f`.
The source/weight/coverage gate is verified and its 22-video TEST inference is
running. Full V3 and head-only no-geometry have completed two selection epochs
and continue their registered training. Four parameter-free controls completed
all-24 TRAIN replay and full row/checksum/candidate-equivalence checks.

The first all-24 no-geometry epoch also has 438 observed preserved clip boundaries,
24 empty history states at true scene origins, finite model parameters, nonzero
birth-pair weight updates and 28,754 optimizer steps. Its intermediate snapshot
is preserved on NAS, separately from the selected final checkpoint. This proves
actual training flow; it is not TEST accuracy evidence.

## Current identity evidence

After selection epoch 1 on the fixed six TRAIN VAL scenes, Full V3 has 4,834
same-person splits, 3,056 different-person merged pairs and 4,160 diagnostic
temporal ID changes. Full no-geometry has 5,865 / 3,232 / 2,664; head-only
no-geometry has 5,790 / 2,071 / 2,878. Thus the current cross-view aggregate
improvement coexists with temporal regression. Independent net geometry benefit
and superiority to GMT are not established.

These counts are not official IDSW, HOTA, AssA, IDF1 or CVIDF1. Official current
TEST columns remain empty until complete outputs and native scoring exist.

The training key `first_appearance_split_count` is defined by the actual target
builder as a split of a model-history-absent same-person birth pair. It is not
restricted to first appearance in the video. Existing raw keys/counts are kept;
the main report label is corrected. Event attribution will separately measure
first clean visibility and absence from prior predicted history.

The existing `teacher_prefix_*` ID counts are measured after the teacher-forced
current assignment correction. They are not the learned solver's current
decisions with only a corrected past. A clean-history causal comparison must
inspect the normal solver before that correction; zero oracle-corrected ID
errors alone do not prove learned model accuracy or isolate exposure bias.

## Frozen R2 birth-utility probes

Nine CPU tests cover eight birth-utility scenarios and a three-method spatial
recovery probe, without real data
or training. With no competing history, a wrong pair just above the registered
threshold can have a strictly positive benefit and beat two zero-benefit
singleton births, even when the singleton prior is extremely small. At exactly
zero benefit the tested solver keeps the singleton baseline. A correct historical
competitor can defeat the weak wrong pair; reliable conflicting geometry can
veto it. Four raw wrong high-score pairs can yield two selected wrong merges,
illustrating why raw Pair FP is not a final false-merge count.

The result is in `results/worldobject_v3_birth_utility_synthetic.json`; tests are
in `analysis/test_worldobject_v3_birth_utility_diagnostic.py`. This establishes a
synthetic structural possibility, not its real TRAIN frequency or causal share
of R2 errors. The real raw-mask/high-score/geometry/TopK/selected/final-ID funnel
and frozen-state counterfactual audit are still pending. A relative joint-versus-
two-singleton utility remains a possible R3 proposal only after TRAIN evidence.

An additional frozen-source probe gives the same person correct shared IDs in
both views after a spatial-conflict visual recovery, then assigns a new shared
ID on the immediately following identical observation. The recovery preserves
the old spatial center but clears `missed` to zero; the next spatial conflict
therefore lacks the recovery exception. Full and head-only geometry reproduce
this behavior while full no-geometry retains the original ID. These are synthetic
world units, not measured physical motion. The real TRAIN frequency, margins and
causal contribution still require event replay; R2 is not patched in place.

## Lifecycle probability semantics

The actual NULL loss uses `sigmoid(null_logit - max_current_assignment_logit)`,
with current score zero when there are no observations. The legacy count
`null_logits >= 0` is not that trained conditional NULL probability. A near-zero
raw activation count alone therefore does not establish lifecycle failure.
Calibration must join this relative probability, clean masked target and actual
solver NULL outcome. Existence supervision also includes the registered retention
window, so a high positive rate must be interpreted against that target policy.
The numerical probability/outcome audit remains pending.

## Input-equivalence audit

The actual old-support-plus-readonly-ROI route has been compared with the new
correct-camera-depth route over all 24 TRAIN scenes, 28,754 paired times and
557,002 candidates. Bbox/score/D1024 ROI inputs are bitwise identical, and the
current GT labels and supervision masks match; see
`results/worldobject_v3_input_equivalence_train.json`. The audit checks scene,
timestamp, camera, detection key, bbox, score, normalized D1024 ROI bytes/tolerance,
current clean GT labels and supervision masks. Geometry is the intentional
difference. Historical labels can differ as a consequence of predicted history;
that is distinct from a current-input or annotation mismatch.

The legacy sparse container does not always store an explicit RGB hash. Both
appearance routes are bound to the same immutable GMT raw cache RGB hashes and
exact image IDs; explicit sparse RGB hashes must also agree. The first audit
attempt assumed that optional legacy field was mandatory and exited before
comparing data. The observer assumption was corrected, the failed attempt is
preserved, and R2 training/cache files were untouched. This is not evidence of
an ROI/data mismatch. The full TEST equivalence pass follows cache completion.

## Checkpoint retention and limits

A separate observer watches atomic `resume.pt` replacements and copies complete
checkpoints with SHA verification/source binding. It sends no training signals,
writes no training files and creates no CUDA context. Full/head-only selection
epochs 1 and 2 are retained; future available epochs are captured as they finish.
The inventory is `results/worldobject_v3_epoch_snapshot_inventory.json`.

No-geometry selection checkpoints 1–4 had already been overwritten before this
addendum. Original aggregate/periodic logs remain, but they cannot reconstruct
all original per-frame decisions or exact per-epoch HOTA/AssA/IDF1. Those results
must remain unavailable unless an authentic historical checkpoint is found.
No retraining or pre-R2 checkpoint is substituted. Final-refit TRAIN replay is
also distinct from held-out selection VAL because the final model has seen all
24 TRAIN scenes.

## Remaining real audits

The newly named birth, temporal, lifecycle, geometry and spatial/temporal JSON
files explicitly mark missing real-event joins as pending. They are not completed
scientific conclusions. Remaining work joins clean TRAIN targets, raw model
probabilities, actual constrained choices and eventual delivered IDs; groups by
scene/camera/gap/density/reliability; checks fusion followed by ID break,
geometry-rejected history, occlusion recovery and cumulative contamination; and
evaluates retained selection checkpoints on complete legal TRAIN VAL videos.

Registered full TEST inference, legal-14 TrackEval/MATLAB, same-device live timing,
negative-result preservation and GitHub delivery continue in parallel. R2 source
and selection rules stay frozen throughout these audits.
