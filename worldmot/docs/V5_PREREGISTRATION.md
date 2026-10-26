# WorldMOT V5 — Observation-Anchored Persistent World Objects

Human authorization: the V5 full research instruction supplied on 2026-10-27
(Asia/Shanghai). Base `eda0761b15f4049ead1971facd17d846bbab94cf`; independent
branch `research/worldmot-v5-decoupled-birth-20261026`. Previous code, weights,
locks, annotations and results stay immutable.

The membership network only predicts historical membership and one missing-
observation option per object per camera. Detector-supported birth, observable
lifecycle, trusted identity-memory writes and C1 Kalman/Joseph updates are
separate deterministic policies. Every observation has an audited decision.
Outputs use current detector boxes only. No history-frame ID rewrites, no
GT-conditioned online decisions and no neural birth/death/write heads.

`configs/worldobject_v5/registered.json` is the finite experiment definition.
Single seed 20261027, identical fresh initialization for B2/B3/B4. Hidden width
256 and geometry width 32 give exactly equal network capacity. No model receives
weights trained on the six validation scenes during selection. All selection
uses the existing fit18 / validation6 partition. Every epoch preserves a complete
scene's predicted history. Gradient accumulation never clears the object bank.
TRAIN supervision fixes each ID's first unambiguous observed GT identity in a
separate loss sidecar. Later wrong historical reuse remains a supervised
negative; it does not erase every loss on that query. Initially mixed/ambiguous
objects are masked. The GT anchor never changes, never enters a feature, and
never repairs the object's appearance, spatial state or runtime ID.

All three trained variants complete 12 selection epochs. The registered B3
TRAIN-VAL convergence rule may extend the **common** budget to 16; otherwise the
budget is 12. Each final model is newly initialized and retrained on all24 scenes
for that common budget. An early selection checkpoint cannot pass the final
TRAIN gate. Training uses AdamW, one epoch warmup and cosine decay. Resume occurs
at verified complete-scene boundaries, retaining optimizer, schedule and RNG.
The schedule reaches its minimum at epoch12; a registered extension continues
at that minimum without a learning-rate restart. Scene order is a same-seed
permutation for every epoch and every trained variant.

Resource plan: use genuinely idle V100s only. When only one GPU is available,
B3 receives it, while B2 and the B4 selection phase may use spare CPU cores.
The final B3/B4 full24 training uses the same V100 FP32 execution conditions;
selection-phase CPU weights are discarded before their fresh final training.
At the actual selection launch all10 GPUs were occupied, so all three selection
jobs started on CPU. To avoid serializing three final jobs behind two available
GPUs, B2's first final launch uses an idle V100 if a third is available, otherwise
one CPU thread. B2 resumes keep that execution family. This resource rule is
fixed before any final training or TEST. The matched B3/B4 geometry comparison
always uses V100 FP32 for both final models; there is no CPU fallback for either.
Actual device, thread count, and timing scope are recorded. No occupied GPU is
claimed merely because it has free memory or temporarily reports zero load.

- B0: original frozen C1, exact reference on unchanged cached inputs.
- B1: frozen C1 **membership/NULL evidence only**, new independent birth and
  explicit lifecycle/memory policies. No old birth/lifecycle/write outputs are
  consumed by the V5 core.
- B2: trained membership with the original 16 geometry features padded to32,
  same independent birth and state policies.
- B3: main, same capacity with quality-controlled world-support, covariance,
  alignment and verified reprojection evidence, bounded uncertainty-aware
  historical geometry, and C1 spatial measurements.
- B4: same capacity, seed, schedule and identity supervision as B3; all explicit
  Ω geometry paths, including birth geometry and spatial measurements, off.

B1−B0 estimates the complete policy separation under the frozen head. Because
disabling the old lifecycle/write controls is explicitly required, it is not
attributed solely to birth. A TRAIN-only same-frozen-history birth counterfactual
isolates birth choices; B1−B0's online error counts are also fully reported.
B2−B1 measures dedicated membership training under the same V5 policies.
B3−B4 is the matched-capacity/budget geometry comparison. B3−GMT is a system
comparison. No extra matched-budget multitask model or seed sweep is registered.

The inherited sparse cache stores foreground support in persistent coordinates
but camera matrices in the current Ω gauge. Its sidecar stores alignment scale,
residual and inlier ratio, **not alignment rotation/translation**. These are not
interchangeable camera frames. Reprojection is enabled only with a verified
world-from-local transform (or the recorded initial-reference identity).
Missing transforms cause explicit reprojection abstention, not fabricated
identity transforms. Existing world-space supports and uncertainty remain
usable. The coverage of every geometry cue is reported, and finite real LIVE
uses the actual current transform. Old caches are not silently reinterpreted or
reconstructed to favorable TEST outcomes.

Birth duplicates require near-identical boxes **and** near-identical ROI
features, with historical explanations audited; crowded overlap alone cannot
suppress a person. Cross-view birth uses support distributions and reliable
projection when available; ambiguous evidence keeps separate tentative IDs.
All valid singleton observations get an immediate ID and current box. High
quality births can be immediately confirmed. The high-quality detector threshold
is calibrated once from fit18 TRAIN clean positives, without discarding frames.

TEST is only opened for inference after selection/budget/configuration and all24
weight gates are frozen. Registered B0–B4 cover all22/29,019 paired times and
548,583 frozen candidates. Native TrackEval and MATLAB use original legal14 GT;
the invalid8 keep complete predictions and separate diagnostics. No TEST model
selection, parameter changes, filtering of raw GT or GT ID repairs. Every
correction is reported with regressions and denominators. Cache throughput,
single-worker latency and actual RGB-to-ID speed have separate labels.

References: TrackTrack `3. Tracker/trackers/tracker.py` and `utils.py` at
`ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da` (track-aware initialization);
[ByteTrack](https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py)
(high/low confidence and observed lifecycle);
[MOTR](https://github.com/megvii-research/MOTR/blob/main/models/qim.py)
(persistent queries and training perturbations). These are design references,
not imported 2D trackers. The [LA-MOTR ICCV2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LA-MOTR_End-to-End_Multi-Object_Tracking_by_Learnable_Association_ICCV_2025_paper.html)
supports separate detection/association; its advertised public GitHub repository
`PenK1nG/LA-MOTR` returned404 on verification, so it is not a code dependency.

The V5 design follows previous TEST-informed V3/V4 analysis: the final TEST is
exploratory, not a previously untouched holdout. Negative results complete the
experiment; they do not authorize a second TEST parameter search.

Scheduling amendment, registered on 2026-10-26 UTC before any V5 TEST inference:
the preceding all-weight-gates-before-inference order is relaxed **only for
already frozen B0 and B1, without GT or scoring**. Their head, configurations,
calibration and complete22 input manifests are already fixed. Three V100s became
idle while the three selection jobs were running normally on CPU. Independent
reference inference can use those cards without restarting training. The subset
authorization permits replay/preflight only; it cannot authorize evaluation,
diagnostics, training, live runs, or B2/B3/B4 inference. B0 scene1 must reproduce
every original C1 timestamp before the reference queue scales out. No TEST GT or
new TEST metric is inspected while model/budget selection is running.

All B2/B3/B4 final-all24 gates are still mandatory before their TEST inference
or **any new formal TEST scoring**. Frozen reference predictions are adopted
into that final scope only with explicit per-scene receipts verifying identical
source, inputs, configurations, original head SHA, original authorization and
immutable output bytes. Their manifests are not rewritten. Posthoc identity
analysis still waits for all110 complete outputs. The original V3/V4 locks and
V5 TRAIN lock stay unchanged. No variant, threshold, seed, loss, training budget
rule, output rule, GT or evaluation setting is changed by this amendment.
