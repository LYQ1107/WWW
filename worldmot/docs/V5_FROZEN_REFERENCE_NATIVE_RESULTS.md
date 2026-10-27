WorldMOT V5 frozen controls: actual full-video native TEST results

Updated UTC: 2026-10-27T07:08:01.333661+00:00.

Each reported control completed all22 videos,29,019 paired times and548,583 frozen candidates. TrackEval and the unchanged VisionTrack MATLAB toolkit scored the original legal14 videos. The other8 were fully inferred and excluded for original GT defects. No GT was cleaned. The early scoring order was published before reading these scores; all B2/B3/B4 training rules remain frozen.

| Method | HOTA | DetA | AssA | IDF1 | CVIDF1 | CVMA | MOTA | IDSW | Frag | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GMT Native (historical) | 66.0646 | 63.0665 | 69.5787 | 80.1121 | 78.0613 | 68.6435 | 72.9636 | 6044 | 1584 | 165338 | 20820 | 24447 |
| V4 C1 (historical) | 59.3712 | 62.9768 | 56.3519 | 67.9896 | 65.8169 | 66.3572 | 74.7746 | 2607 | 1561 | 165338 | 20820 | 24447 |
| B0 | 59.3712 | 62.9768 | 56.3519 | 67.9896 | 65.8169 | 66.3572 | 74.7746 | 2607 | 1561 | 165338 | 20820 | 24447 |
| B1 | 57.5119 | 63.1728 | 52.6809 | 65.1046 | 55.8643 | 43.0898 | 71.4129 | 8973 | 1601 | 165331 | 20827 | 24454 |

B1 minus newly scored B0: HOTA -1.8593, AssA -3.6710, IDF1 -2.8850, CVIDF1 -9.9526, CVMA -23.2674, MOTA -3.3617 percentage points; IDSW +6366, Frag +40.

This is a negative aggregate result for replacing the old policies while retaining the old identity head. B1 also changes history assignment, lifecycle, spatial and appearance-write policies, so the full delta cannot isolate Birth alone. B2/B3/B4 use a newly trained dedicated membership head; their complete results remain pending. These control scores do not establish the independent value of Omega geometry.

B1 IDF1 minus CVIDF1 is 9.2402 points; the reference gap is 2.1727. Together with the CVIDF1 regression, this is consistent with poorer cross-camera identity agreement. It does not identify exact false births, wrong reuses or recovery failures. The registered all110-prediction posthoc analysis will count those events after final model TEST.

The frozen detections were preserved. Small CLEAR TP/FP/FN differences can arise while the candidates remain identical; these results are not evidence of a detector change. CLEAR prioritizes prior identities and HOTA uses global identity alignment before matching. A generated identical-boxes check reproduces changes in CLEAR counts and DetA from IDs alone; it does not locate the real seven events. See V5_IDENTITY_CONDITIONED_DETECTION_COUNTS.md. TrackEval IDSW is the switch metric in the table. VisionTrack's interleaved-camera CLEAR switch count has a different definition and must not replace it.

CVIDF1 improves in 2/14 scenes and regresses in 12/14. Per-scene aggregates below do not alter model selection.

| Scene | Delta HOTA | Delta AssA | Delta IDF1 | Delta CVIDF1 | Delta IDSW |
|---|---:|---:|---:|---:|---:|
| 00001garden | -2.6160 | -5.3782 | -3.8177 | -6.6427 | +484 |
| 00003garden | +1.0881 | +1.9451 | +1.1733 | -1.6866 | +787 |
| 00005garden | -4.8936 | -9.8556 | -8.0140 | -9.0182 | +937 |
| 00012night | +8.8749 | +16.9825 | +19.1710 | -16.2694 | +38 |
| 00019court2 | +7.5712 | +13.6287 | +11.5129 | -5.7275 | -118 |
| 00022gate | -1.0011 | -2.0553 | -3.5628 | -4.4825 | +102 |
| 00023court4 | -22.8920 | -39.7485 | -35.4576 | -38.3341 | +1008 |
| 00027square | -10.9771 | -17.0420 | -18.5430 | -28.5714 | +562 |
| 00028path | +10.2233 | +15.3401 | +20.7387 | +6.3169 | +88 |
| 00029path | +7.8424 | +13.5450 | +9.5133 | -5.9280 | +92 |
| 00117wood | +0.4238 | +0.6345 | +3.8046 | +0.7698 | +1300 |
| 00120park | +10.2551 | +16.8065 | +11.6406 | -4.9785 | +77 |
| 00144road | -1.5149 | -3.6008 | -2.4591 | -12.7817 | +538 |
| 00146bridge | -2.8529 | -5.2015 | -9.7734 | -8.5361 | +471 |

The combined metric is the native combined result, not an unweighted mean of this table.

B0 metrics SHA-256: `9d854785ba039048a90c6ff43e809f8b10b0d2c5959ff9b2ea9225d3eeae2ca2`.

B1 metrics SHA-256: `af5b4b872862b4628e99405a5d21d772930516a4952b2d3933dd6896efcf74fe`.

Both controls use frozen R2 head SHA-256 `cfda40050e89b31dd4480ca4d5010e23067a94ba18ea7d0e2425d5935d3f7fb8`. No optimizer, checkpoint selection, new-model TEST or TEST parameter search is authorized by the control-only scoring scope. Cached replay FPS is reported separately; real RGB-to-ID timing is pending.
