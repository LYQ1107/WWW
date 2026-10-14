# Preregistration — frozen before baseline interventions and new gradients

2026-10-14; seed20261009; source04324354e9e9971eec6d937b961590a0cdbf13a9. Main control is full Original GMT using local frozen Stage2 checkpoint, official VISION_test configuration and its actual weight-compatible RPCE/GTA dimensions, without identity-backbone substitution or dormant GTA pruning. Any necessary loading correction is recorded before primary comparisons.

Order: baseline source/checkpoint audit; short exact unmodified-vs-adapter-OFF parity; complete TRAIN16 exact prediction/state parity and matched official metrics; baseline current-state opportunities on TRAIN12/13/14 first256 frames; zero-training threshold/rule controls; only if certified rule netN01−N10>0 with no known-row capacity errors, matched MLP/Attention/CIC300 updates, upgrade all finite nonzero-gradient qualified arms to fixed cumulative1000 updates. No metric-sign/DEV/step selection.

All learned arms share lawful GMT score-stage inputs, unique certified labels, preserve negatives, optimizerAdamW(lr1e-4,wd0.01), batch4, same source-event schedule, seeds and final-budget checkpoint. At least half sampling slots are broad protection examples. Ambiguous aliases and unsupported histories are UNKNOWN, masked. Native assignment and base thresholds remain the executor; one shared same-solver association pass, no extra JEV-only solve. Maximum one minimal adjustment if ordinary controls improve but CIC has a recorded explainable structural gap.

Rule residual upper bound0.05 on normalized/averaged GMT score units, gate requires >=3 other-view committed observations and observed conflict, row competition margin<=0.10. Fixed threshold control uses original threshold+0.02 only on such frozen risk rows; no TRAIN16 calibration. Scores outside legal candidates unchanged. These constants and all feature semantics will be fixed in executable protocol before running controls.

Train sample acquisition is in-memory on complete native GMT prefixes; no large persistent corpus. Temporal block(frame//64)%5==4 excludes training. Full TRAIN16 runs use actual mutated states and the same GMT inputs/weights/official TrackEval+MATLAB conventions. N01/N10 certificates are distinguished from whole-plan oracle bounds; any oracle is labelled ORACLE, never a deployed result.

Main development success: HOTA>=GMT+0.5, AssA/CVIDF1 no decrease, IDSW/Frag/error duration<=1.05x; sameGPU three full-image trial meanFPS>=0.95x. Extra trainable parameters<100k. Missing independent validation remains pending regardless of development signs. Architecture ablations only after a joint positive main result.

Budget: new12 GPU-wall-hours, total new retainedworktree+runtime<=512MiB, home reserve>=30GiB. data1 read-only due its27GiB free. No new dataset/weights download, no GMT gradients, no DEV or sealed/TEST/Full24 access, preserve historical checkpoints/evidence.
