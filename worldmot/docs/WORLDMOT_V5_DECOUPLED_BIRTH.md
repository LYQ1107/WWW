WorldMOT V5: decoupled observation birth and persistent membership

Updated UTC: 2026-10-26T22:15:57.404794+00:00. Status: **REGISTERED_RESEARCH_IN_PROGRESS**.

The finite B0–B4 registration, code fingerprint and original V3/V4 artifacts are frozen. The model predicts membership and a per-camera no-observation option. Birth, lifecycle, identity writes and C1 Kalman/Joseph updates use separate auditable policies. GT never repairs the predicted history.

Source SHA-256: `c4f9d1c0b2a290062aba52ae7c0d322176ae5376f9c85688f74d89159e16f733`.

All24 TRAIN inputs are verified: 28,754 paired times and 557,002 detections. Fit18/validation6 selection uses 12 epochs with the fixed common 16-epoch extension rule. Final weights start fresh and must prove all24 coverage in every epoch.

| Variant | Selection completed epochs | Current selection scene count | Final completed epochs | Final weight SHA-256 |
|---|---:|---:|---:|---|
| B2 | 3 | 11/18 | 0 | not available |
| B3 | 3 | 7/18 | 0 | not available |
| B4 | 4 | 12/18 | 0 | not available |

Latest completed selection epochs below are TRAIN diagnostics. Loss is averaged over paired timestamps; scene density and evolving predicted histories differ between fit18 and VAL6. The per-scene breakdown is in v5_training_curves.json. These values do not establish convergence or formal TEST improvement.

| Variant | Epoch | Fit membership loss | VAL membership loss | VAL same-person split rate | VAL different-person false-merge rate | Finite gradients |
|---|---:|---:|---:|---:|---:|---|
| B2 | 3 | 0.0693 | 1.7595 | 31.74% | 1.00% | True |
| B3 | 3 | 0.0724 | 1.7671 | 34.36% | 0.79% | True |
| B4 | 4 | 0.0586 | 1.1919 | 33.34% | 0.45% | True |

Selection currently uses CPU, one Torch/BLAS thread per worker. Final B3/B4 training and complete TEST use truly idle V100 FP32 workers. B2 final training uses a third idle V100 if available, otherwise one CPU thread; its resumes keep that device family. Other users' processes are never stopped. Independent-scene concurrency is not single-stream FPS.

A preregistered scheduling amendment allows already frozen B0/B1 GT-free reference inference during TRAIN selection. Its separate authorization cannot score TEST or run a new trained model. All three final-all24 gates remain required before new-model TEST or any formal scoring. Reference outputs require explicit byte-verified scope-adoption receipts before final scoring. No current model, budget rule or input is changed.

Official MC-MOT uses original legal14 full TEST videos, after full22 causal inference (29,019 paired times / 548,583 shared candidates). Invalid8 remain separate. The following historical reference numbers are not new V5 measurements.

Native scorer engineering smoke: three generated synthetic cases and six native MATLAB jobs passed. No real TRAIN/TEST data or model was used. The native toolkit hashes exactly match archived V4 C1. This is evaluator validation, not a V5 tracking score.

| Method | HOTA | DetA | AssA | IDF1 | CVIDF1 | CVMA | MOTA | IDSW | Frag | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GMT Native | 66.0646 | 63.0665 | 69.5787 | 80.1121 | 78.0613 | 68.6435 | 72.9636 | 6044 | 1584 | 165338 | 20820 | 24447 |
| GMTDet + V2 | 61.0177 | 62.1099 | 60.2274 | 74.2189 | 72.6929 | 68.3068 | 75.4564 | 598 | 1289 | 153813 | 10010 | 35972 |
| Full No-Geometry V3 | 60.4318 | 63.0806 | 58.2867 | 70.1622 | 67.6049 | 64.1136 | 75.0228 | 2130 | 1575 | 165335 | 20823 | 24450 |
| Full V3 + Omega | 56.4218 | 63.1085 | 50.7806 | 63.9655 | 61.8413 | 66.3651 | 74.6619 | 2837 | 1566 | 165346 | 20812 | 24439 |
| V4 C1 | 59.3712 | 62.9768 | 56.3519 | 67.9896 | 65.8169 | 66.3572 | 74.7746 | 2607 | 1561 | 165338 | 20820 | 24447 |
| B0 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B1 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B2 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B3 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| B4 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

TRAIN conditional birth audit: 6/6 complete scenes, 5860 paired times. Only original C1 decisions update history; alternative births are discarded after comparison. These are TRAIN clean-label diagnostic events, not HOTA or CVIDF1.

INDEPENDENT_BIRTH_FIXED_ORIGINAL_HISTORY: same-person pair corrections 0, regressions 20, net -20; different-person pair net +0; false-birth avoidance net +0.

LEGACY_HISTORY_ONLY_THEN_INDEPENDENT_BIRTH: same-person pair corrections 8, regressions 20, net -12; different-person pair net +0; false-birth avoidance net +0.

Independent frozen C1/B1 TRAIN closed-loop audit: 6/6 complete scenes, 5860 paired times. Each method advances its own predicted history with no optimizer. The frozen head was trained on these scenes, so these are mechanistic TRAIN diagnostics, not held-out TEST scores. This audit cannot alter the registered membership model or its training budget.

Frozen online B1 minus C1: same-person pair corrections 206, regressions 18902, net -18696; different-person separation net +422. Clean global one-to-one overlap net -20908 observations, not CVIDF1.

B1−B0 also changes explicit lifecycle and memory-write policy, so its full online delta cannot be assigned solely to Birth. B2−B1 includes a new trained head and capacity/budget changes. B3−B4 is the equal-capacity, equal-budget explicit-geometry comparison.

The inherited cache has world-space person supports but lacks the rotation/translation needed to move its local camera matrices into that same world gauge. Reprojection explicitly abstains except at verified reference transforms. Fit18 calibration found verified cameras at 18/22,894 paired times. This limits what B3 can establish about reprojection. New real LIVE timing uses actual current transforms and is not a full-video accuracy comparison.

Cached replay speed and real RGB-to-final-ID speed are separate. The finite real timing scope is TRAIN scenes4/8/12, first61 paired times each, after5 warm-up times and a state reset. It includes actual RGB read, detector, per-frame Ω (except B4), alignment, membership, C1 updates and final ID serialization/flush. It is not full TEST RGB-to-ID FPS.

| Method | Cached worker step FPS | Cached worker wall FPS | Real RGB-to-ID FPS | Real scope |
|---|---:|---:|---:|---|
| B0 | 33.6722 | 13.0486 | not measured | NOT_MEASURED |
| B1 | 17.4946 | 16.8607 | not measured | NOT_MEASURED |
| B2 | not measured | not measured | not measured | NOT_MEASURED |
| B3 | not measured | not measured | not measured | NOT_MEASURED |
| B4 | not measured | not measured | not measured | NOT_MEASURED |

Native tool dependencies and wrapper comparability are retained from the archived V4 report. Historical GMT/V2 exports used two-decimal coordinates; current V3/V4/V5 exports preserve .17g. Original GT and shared native tracking dependencies were verified in the archived comparison. This is a documented system-reference limitation, not permission to alter GT or re-tune TEST.

V5 research remains in progress. No V5 improvement claim is made before the registered full training and formal TEST finish.

Recovery update 2026-10-26T21:53:02.754725+00:00: B0 full22 exact C1 reproduction is complete; frozen B1 inference continues and all3 TRAIN selection runs remain active. See V5_STORAGE_RECOVERY.md and v5_B0_full22_reproduction.json. No new formal TEST score is available. Cached timing snapshot above predates recovery; mixed NAS/local output timing cannot rank algorithm speed.

Frozen reference completion 2026-10-26T22:15:57.404794+00:00: B0 and B1 each finished22/22 TEST videos,29,019 paired times and548,583 frozen candidates. B0 exactly reproduced original C1 at every time. B1 retained all548,583 current detection boxes, with0 duplicate suppression and one state reset at each scene origin. These GT-free checks do not measure B1 identity accuracy. Native metrics remain pending until the three final full24 TRAIN gates. See v5_frozen_reference_inference.json.

B1 cached replay timing now covers all22 videos; B0 cached timing above still covers its earlier12-video snapshot. The runs have mixed NAS/local original output paths and do not provide a controlled algorithm speed ranking or real RGB-to-ID FPS.
