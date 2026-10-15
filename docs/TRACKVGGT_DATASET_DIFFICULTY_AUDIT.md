# VisionTrack original TRAIN dataset difficulty census

CPU-only census of all24 original TRAIN sequences. All48 raw GT hashes match the existing uniqueness audit, whose duplicate/conflict counts are reused. No VGGT, tracking replay, TEST data or GT repair. Original valid active class1 rows, including duplicates, are counted; raw identities in conflicting sequences remain descriptive.

|Scene|Role|Groups|GT rows|Raw identities|Multi-camera identities|Patch0 /1–3 /4–15 /16+|Conflicting groups|
|---|---|---:|---:|---:|---:|---|---:|
|1 00002garden|strict_DEV5|932|4395|10|2|3 /3 /238 /4151|0|
|2 00004garden|conflicting_GT_DEV_diagnostic|1886|13998|11|8|0 /1339 /7635 /5024|1|
|3 00006garden|conflicting_GT_DEV_diagnostic|1200|37703|32|14|7037 /21191 /7739 /1736|1|
|4 00014night|association_fit|901|3840|4|2|197 /1956 /821 /866|0|
|5 00016court1|association_fit|1769|6916|3|2|1 /5 /5335 /1575|0|
|6 00017court1|association_fit|601|2573|3|2|0 /108 /1924 /541|0|
|7 00020court3|association_fit|443|1772|2|2|7 /0 /1756 /9|0|
|8 00021gate|association_fit|523|2092|2|2|0 /0 /1860 /232|0|
|9 00024court4|association_fit|1942|8720|3|2|0 /515 /7105 /1100|0|
|10 00025square|strict_DEV5|2092|14092|4|4|411 /9613 /4068 /0|0|
|11 00026square|strict_DEV5|2057|15142|6|4|509 /11023 /3610 /0|0|
|12 00030path|association_validation|1200|14867|15|8|3678 /8195 /2942 /52|0|
|13 00031path|association_fit|1621|17099|10|8|3567 /9983 /3495 /54|0|
|14 00042football|conflicting_GT_DEV_diagnostic|969|35039|27|21|14333 /19512 /1194 /0|582|
|15 00050football|conflicting_GT_DEV_diagnostic|1048|85437|73|36|35282 /47647 /2508 /0|16|
|16 00066football|conflicting_GT_DEV_diagnostic|1034|59902|55|19|30975 /26689 /2238 /0|852|
|17 00116wood|association_fit|1200|11731|8|3|1976 /6721 /2744 /290|0|
|18 00118park|strict_DEV5|1029|11769|12|5|7068 /3935 /766 /0|0|
|19 00119park|strict_DEV5|1052|16285|15|9|7630 /7133 /1506 /16|0|
|20 00145road|association_fit|1001|26282|26|14|14472 /10213 /1527 /70|1|
|21 00147bridge|association_fit|1001|8026|11|5|2572 /4227 /1177 /50|0|
|22 00155basketball|conflicting_GT_DEV_diagnostic|1000|21364|12|11|1352 /12828 /7184 /0|71|
|23 00160basketball|conflicting_GT_DEV_diagnostic|1052|99389|59|49|47968 /50226 /1195 /0|1|
|24 00182canteen|association_fit|1201|78316|76|26|43412 /30438 /4466 /0|366|

Patch coverage uses the actual518 pad mapping and14-pixel centers. A zero-center box may still support ROIAlign. GT size statistics cannot establish detector failures, appearance discriminability, or candidate headroom; those require the causal audit. Missing annotation count is NOT_MEASURED. Empty labeled frames are reported separately.

Locked Cohort-V2 verification: 1000 fit /300 validation, all raw artifacts and2600 token files hash-verified; 502195973 bytes. Four extraction queues finished with all stages PASS. No teacher training has started.

## Later research status

The data conclusions above are retained from the pre-feature/pre-training audit
phase. Their NOT_MEASURED/pending teacher fields describe that snapshot. Current
TrackVGGT research has completed all56 registered teacher runs, the V2/V3
comparison, crop/QD/temporal controls and the negative relation gate. See
TRACKVGGT_TEACHER_RESULTS.md and TRACKVGGT_FINAL_REPORT.md; no frozen audit
statistics, raw inputs or original result JSON is rewritten by this note.
