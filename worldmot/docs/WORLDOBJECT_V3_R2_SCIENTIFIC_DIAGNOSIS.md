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
The source/weight/coverage gate is verified; its 22-video TEST inference has
completed, with native TrackEval/MATLAB scoring and a true live sample.
Full V3 and head-only no-geometry completed four selection epochs. The unchanged
TRAIN-only rule selected final budgets of one and three epochs respectively.
Full finished its all24 final epoch, with checkpoint SHA-256
`cfda40050e89b31dd4480ca4d5010e23067a94ba18ea7d0e2425d5935d3f7fb8`;
all22 TEST inference and legal14 native scoring are complete. Head-only
continues its normal three-epoch final training. Four parameter-free controls completed
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

These TRAIN counts are not official IDSW, HOTA, AssA, IDF1 or CVIDF1. Full
no-geometry now has official legal-14 HOTA 60.4318, AssA 58.2867, IDF1 70.1622
and CVIDF1 67.6049, below GMT and V2. MOTA 75.0228 exceeds GMT 72.9636 while
identity scores remain lower. Full's formal HOTA/AssA/IDF1/CVIDF1 are now
56.4218/50.7806/63.9655/61.8413, all below No-Geometry. The registered Full-vs-
NoGeometry contrast therefore shows no independent geometry gain. Head TEST
remains pending normal training; no parameter or budget is changed using TEST. See
`results/worldobject_v3_no_geometry_official_test.json` for hashes and live timing.

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
of R2 errors. The first real retained-checkpoint funnel is measured below;
later epochs and the frozen-state counterfactual remain pending. A relative joint-versus-
two-singleton utility remains a possible R3 proposal only after TRAIN evidence.

An additional frozen-source probe gives the same person correct shared IDs in
both views after a spatial-conflict visual recovery, then assigns a new shared
ID on the immediately following identical observation. The recovery preserves
the old spatial center but clears `missed` to zero; the next spatial conflict
therefore lacks the recovery exception. Full and head-only geometry reproduce
this behavior while full no-geometry retains the original ID. These are synthetic
world units, not measured physical motion. The first real TRAIN frequency and
margins are measured below; R2 is not patched in place.

## Lifecycle probability semantics

The actual NULL loss uses `sigmoid(null_logit - max_current_assignment_logit)`,
with current score zero when there are no observations. The legacy count
`null_logits >= 0` is not that trained conditional NULL probability. A near-zero
raw activation count alone therefore does not establish lifecycle failure.
Calibration must join this relative probability, clean masked target and actual
solver NULL outcome. Existence supervision also includes the registered retention
window, so a high positive rate must be interpreted against that target policy.
The first retained-checkpoint probability/outcome audit follows; later available
checkpoints are still being replayed.

The first four real retained-checkpoint replays are now complete: Full and
Head-only selection epochs 1–2, all six fixed TRAIN VAL scenes / 5,860 synchronous
times / 172,760 current detections each. All four exactly reproduce every comparable
original epoch diagnostic count; the observer changes no decisions or state.
Epochs 3–4 are still being replayed. The detailed numbers below
refer to these **selection** weights, not the final all24-refitted TEST weights.

| Actual TRAIN VAL event | Full epoch 1 | Head-only epoch 1 |
|---|---:|---:|
| Correct selected spatial-conflict visual recoveries | 6 | 7 |
| Same person cleanly detected immediately next frame | 6 | 7 |
| Any next-frame change away from the recovered ID | 6 | 7 |
| Next-frame actual new ID | 1 | 1 |
| ID break with correct history actually geometry-rejected | 5 | 6 |
| New ID with retained stale center and geometry rejection | 1 | 1 |
| High-score DIFFERENT-person pair candidates | 13,233 | 19,194 |
| Selected DIFFERENT-person joint Births | 0 | 1 |
| Final DIFFERENT-person merged pairs, all paths | 3,056 | 2,071 |

All recovery events in this first pass occur in Scene 24's clean annotation
subset. Its raw GT is not legal for official scoring. Most next-frame changes
reuse another existing ID rather than creating a new one. Thus the exact
synthetic stale-position/new-ID mechanism does occur in real TRAIN, but these
small counts do not establish that it explains most temporal errors.
Full's Scene 24 timestamp 704→705 is one concrete instance: recovered ID 158
keeps its old center and resets missed=0; next-frame appearance cosine 0.9743
and learned identity margin 4.9267 still lose to spatial distance 11.4044
against gate 6, and the affected detection receives new ID 162.

Full's actual trained NULL-margin probability has 8,169 values ≥.5 despite
zero nonnegative raw NULL logits. Its mean probability is .04821 versus a
positive-target fraction .07172, Brier .04145 and ten-bin ECE .02675.
Existence averages .99954 against .99922 positive targets under the retention
policy. These measured target/probability results replace claims of lifecycle
collapse inferred from raw-logit counts alone. The JSONs also report raw Birth
and history-relative softmax calibration, with exact sample counts and bins.

Unchanged TrackEval on complete legal TRAIN VAL scenes 4/8/12 yields Full
HOTA 74.7869 / AssA 71.7914 / IDF1 85.2940 and Head-only 75.9365 / 74.0132 /
86.4258. The other three fixed VAL scenes 16/20/24 are excluded from that
official protocol without editing GT. These are TRAIN VAL metrics and cannot
be compared with GMT's legal14 TEST aggregate as an improvement claim.

Epoch 2 separately gives Full 6 correct spatial recoveries, 1 next-frame new ID,
and 5 ID breaks with actual geometry rejection; Head-only gives 7 / 2 / 6.
These counts are repeated evaluations of the same VAL videos with different
weights, so they must not be pooled as independent real-world events.

Evidence is in `results/worldobject_v3_retained_TRAIN_VAL_summary.json` and the
birth, lifecycle, geometry and spatial/temporal attribution JSONs. They explicitly
say four of eight requested retained replays are complete at this snapshot.
A separate R3 branch, `research/worldmot-v3-r3-state-update-fix-20261025`, prepares
spatial abstention after accepted visual recovery. Its package plus three new
regressions pass 203 CPU tests. One fixed-weight TRAIN-only replay is registered
after diagnostic GPU release; no R3 optimizer or long training is started.

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
an ROI/data mismatch. The full TEST equivalence pass also passed all22 / 29,019
times / 548,583 candidates with bitwise-identical bbox/score/ROI and no TEST GT
access; see `results/worldobject_v3_input_equivalence_test.json`.

## Checkpoint retention and limits

A separate observer watches atomic `resume.pt` replacements and copies complete
checkpoints with SHA verification/source binding. It sends no training signals,
writes no training files and creates no CUDA context. Full/head-only selection
epochs 1–4 are all retained and SHA-bound to their original training boundaries.
The inventory is `results/worldobject_v3_epoch_snapshot_inventory.json`.

No-geometry selection checkpoints 1–4 had already been overwritten before this
addendum. Original aggregate/periodic logs remain, but they cannot reconstruct
all original per-frame decisions or exact per-epoch HOTA/AssA/IDF1. Those results
must remain unavailable unless an authentic historical checkpoint is found.
No retraining or pre-R2 checkpoint is substituted. Final-refit TRAIN replay is
also distinct from held-out selection VAL because the final model has seen all
24 TRAIN scenes.

## Remaining real audits

The birth, temporal, lifecycle, geometry and spatial/temporal JSON files contain
real event joins for the completed retained replays and mark remaining epochs
as incomplete. Remaining work completes all available retained checkpoints,
the separate TRAIN-only R3 counterfactual, and full TEST geometry/Birth contrasts.
No-Geometry's completed all22 posthoc TEST attribution is documented separately
in `WORLDOBJECT_V3_R2_NO_GEOMETRY_IDENTITY_ATTRIBUTION.md`; it does not choose R2
parameters or substitute for the Full-vs-NoGeometry formal TEST comparison.

Registered full TEST inference, legal-14 TrackEval/MATLAB, same-device live timing,
negative-result preservation and GitHub delivery continue in parallel. R2 source
and selection rules stay frozen throughout these audits.
