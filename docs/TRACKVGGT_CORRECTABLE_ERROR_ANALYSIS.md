# Correctable identity error analysis

Status: COMPLETE_12_ASSOCIATION_TRAIN_FULL_CAUSAL_AUDITS. Fit totals exclude validation scene12.

|Scene|A ranking|B actual known failure|Threshold rejected (all)|No pure candidate|Known mixed history|Unknown reference observation|Known local NULL|GT not detector-matched|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|4|24|242|301|100|428|2803|0|281|
|5|0|52|62|33|0|1331|0|2|
|6|0|29|43|39|0|1525|0|0|
|7|8|63|72|135|339|639|0|77|
|8|6|25|33|78|758|1027|0|121|
|9|10|171|186|78|215|2009|0|40|
|12|20|23|89|1886|7143|12885|3|715|
|13|11|27|72|1104|3049|8181|2|298|
|17|17|41|65|977|2201|8763|2|724|
|20|102|148|321|3643|20844|22515|7|2642|
|21|2|6|72|1333|6967|4764|1|1070|
|24|420|672|1646|20077|69971|69993|0|9538|

These columns are not mutually exclusive. Threshold rejection is an actual solver observation; NULL dominance and multiplicity-boundary evidence are separate overlapping flags, not invented exclusive causes. No pure candidate does not identify a causal detector failure. GT not detector-matched is a separate one-to-one detector coverage statistic and cannot be added to query errors.

Among known fit primary errors with a full pure positive, Top8 retains 1312 /1476; this differs from overall candidate coverage dominated by easy queries. A known correct observation inside an impure or partially unlabelled reference track cannot certify a pure candidate. The unlabelled-history policy can remove otherwise correct real identities from this conservative supervised denominator, so missing-pure counts are not all proven retrieval failures.

|518 patch centers|Queries|Known wrong/missed|Correctable /known wrong|GTA Recall1 /pure-positive-full|VGGT ROI Recall1|DINO Recall1|
|---|---:|---:|---|---|---|---|
|0|56860|540|511 /540|30969 /35354|NOT_MEASURED|NOT_MEASURED|
|1-3|62619|390|369 /390|41088 /45611|NOT_MEASURED|NOT_MEASURED|
|4-15|31376|372|371 /372|27126 /28546|NOT_MEASURED|NOT_MEASURED|
|16+|4056|225|225 /225|3806 /3942|NOT_MEASURED|NOT_MEASURED|

Completed fit sequences contain 1619 Top8-reachable known ranking or actual primary wrong/missed queries; 1489 lie outside fixed V2. Reachable pure-known Top1 errors: 514. These overlap actual solver errors and can repeat along one identity. 578 adjacent-group pairs occur within the same scene/camera/identity error history; do not equate queries with independent error episodes.

DEV5 retrospective is saved separately. It cannot determine fit selection or AG/QD architecture. Scene1 saved actual solver evidence cannot establish candidate coverage in the other four scenes, where those fields remain NOT_MEASURED.

<!-- SEALED_DATA_CONCLUSIONS -->

Fit primary correctable errors with any other-camera pure positive: 899; with only other-camera pure positives: 54. Same-camera support often coexists, so these are not all exclusively cross-camera failures.

Of 27597 fit queries without a pure positive, 27470 still contain the correct GT observation in impure/unreliable reference tracks. This is conservative supervised candidate availability, not proof of 27597 detector failures or absent correct observations.

|Cohort|Unique fit queries|Scenes|Known scene/identities|Reachable ranking or primary errors|Pure-Top8 ranking errors|Genuine legal-history NULL|
|---|---:|---:|---:|---:|---:|---:|
|V2|1000|5|11|130|19|0|
|V3|3000|11|133|1068|302|12|

V3 includes 2721 pure-Top8-positive queries and 12 genuine legal-history NULL queries; the other 267 queries retain supervision masks. Zero-center queries: 565/3000; crop/ROI controls remain necessary.

A single pre-feature/pre-training index correction reserved rare genuine NULL before the abundant missing/unknown pool. The first V3 index and comparison remain archived by exact hash in /data1/liuyeqiang/WWW_trackvggt_20261015/manifests/trackvggt_expanded_training_v3_initial_index_v1.json. The final revision adds no duplicate query and all added known queries respect the3-group spacing from both V2 and each other. Original V2 is unchanged. Teacher V2/V3 benefit is NOT_MEASURED.

## Later research status

The data conclusions above are retained from the pre-feature/pre-training audit
phase. Their NOT_MEASURED/pending teacher fields describe that snapshot. Current
TrackVGGT research has completed all56 registered teacher runs, the V2/V3
comparison, crop/QD/temporal controls and the negative relation gate. See
TRACKVGGT_TEACHER_RESULTS.md and TRACKVGGT_FINAL_REPORT.md; no frozen audit
statistics, raw inputs or original result JSON is rewritten by this note.
