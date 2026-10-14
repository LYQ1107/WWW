# Preregistration — frozen before baseline interventions and new gradients

2026-10-14; seed20261009; source04324354e9e9971eec6d937b961590a0cdbf13a9. Main control is full Original GMT using local frozen Stage2 checkpoint, official VISION_test configuration and its actual weight-compatible RPCE/GTA dimensions, without identity-backbone substitution or dormant GTA pruning. Any necessary loading correction is recorded before primary comparisons.

Order: baseline source/checkpoint audit; short exact unmodified-vs-adapter-OFF parity; complete TRAIN16 exact prediction/state parity and matched official metrics; baseline current-state opportunities on TRAIN12/13/14 first256 frames; zero-training threshold/rule controls; only if certified rule netN01−N10>0 with no known-row capacity errors, matched MLP/Attention/CIC300 updates, upgrade all finite nonzero-gradient qualified arms to fixed cumulative1000 updates. No metric-sign/DEV/step selection.

All learned arms share lawful GMT score-stage inputs, unique certified labels, preserve negatives, optimizerAdamW(lr1e-4,wd0.01), batch4, same source-event schedule, seeds and final-budget checkpoint. At least half sampling slots are broad protection examples. Ambiguous aliases and unsupported histories are UNKNOWN, masked. Native assignment and base thresholds remain the executor; one shared same-solver association pass, no extra JEV-only solve. Maximum one minimal adjustment if ordinary controls improve but CIC has a recorded explainable structural gap.

Rule residual upper bound0.05 on normalized/averaged GMT score units, gate requires >=3 other-view committed observations and observed conflict, row competition margin<=0.10. Fixed threshold control uses original threshold+0.02 only on such frozen risk rows; no TRAIN16 calibration. Scores outside legal candidates unchanged. These constants and all feature semantics will be fixed in executable protocol before running controls.

Train sample acquisition is in-memory on complete native GMT prefixes; no large persistent corpus. Temporal block(frame//64)%5==4 excludes training. Full TRAIN16 runs use actual mutated states and the same GMT inputs/weights/official TrackEval+MATLAB conventions. N01/N10 certificates are distinguished from whole-plan oracle bounds; any oracle is labelled ORACLE, never a deployed result.

Main development success: HOTA>=GMT+0.5, AssA/CVIDF1 no decrease, IDSW/Frag/error duration<=1.05x; sameGPU three full-image trial meanFPS>=0.95x. Extra trainable parameters<100k. Missing independent validation remains pending regardless of development signs. Architecture ablations only after a joint positive main result.

Budget: new12 GPU-wall-hours, total new retainedworktree+runtime<=512MiB, home reserve>=30GiB. data1 read-only due its27GiB free. No new dataset/weights download, no GMT gradients, no DEV or sealed/TEST/Full24 access, preserve historical checkpoints/evidence.

## Executable details frozen before P1 outcomes

24 observable features and feature units are defined in gtr/modeling/gmt_jev/correction.py. Risk requires >=2 lawful candidates, top-two normalized GMT margin<=0.10, >=3 committed other-view observations, absolute appearance support conflict>=0.05. Rule delta=0.05*clip((other−same)/0.2,−1,1); fixed calibration delta=−0.02. Residuals multiply the original support count when added to summed native scores; native0.1 overlap and0.4 Bank thresholds remain unchanged. MLP5119/Attention4898/CIC4962 parameters. Every model receives the entire same24-feature candidate matrix; MLP uses ordinary shared/row means, Attention uses shared context attention, CIC additionally uses2 observable conflict-question roles.

Evaluation conventions: primary unfiltered native online predictions expose all fragments and wrong births; stock GMT min_track_len50 plus unchanged native postprocessing is a matched secondary, reported separately for both methods. Neither convention may be selected by its metric sign. Raw and stock metrics are never mixed. Complete image FPS includes the stock final postprocessing. Source-state ORACLE diagnoses are labelled non-deployable and do not count as primary tracking results.

Training coefficients if qualified: native-association CE on unique legal targets, preserve penalty1.0 on normalized residuals for certified already-correct GMT rows, residual penalty0.01 on all normalized residuals; AdamW1e-4/0.01,300 then1000 cumulative, fixed final checkpoint. Native assignment measurements and full closed-loop tracking decide the result, rather than pairwise accuracy.

目标唯一性补充（P1结果前冻结）：其他ID即使尚未达到3次认证，只要已有当前人物的纯净支持且纯度>=80%，也作为可能别名导致UNKNOWN。不能把未认证短ID伪装成错误负例。Bank仅统计原生实际查询和接受的选项，不从陈旧的可能成员表虚构合法候选。

## One bounded Case-B calibration probe after P1 (no learned update)

P1 found25 incorrect certified-active rows, original cross-rule0/0. One new conservative rule diagnostic: residual=−clip(0.1−GMT_average,0,0.05), preserving all already-acceptable edge scores. Analyze first camera link separately with pure observable relative cosine residual0.05*clip((other_cos−rowmean_other_cos)/0.1,−1,1), requiring low GTA margin<=0.10 and cosine>=0.8. Both use identical current native states/candidates and the original Hungarian/thresholds. First-frame GT IoU>=0.8/runner-up<0.3 bridge is offline diagnosis only, not a long-term owner or future-safety label. A positive same-state signal still needs actual native zero-training confirmation before any MLP/CIC gradients. No video16 calibration or hyperparameter search.
