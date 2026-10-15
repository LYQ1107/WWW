# Correctable identity error analysis

Status: INCOMPLETE. Fit totals exclude validation scene12.

|Scene|A ranking|B actual known failure|Threshold rejected (all)|No pure candidate|Impure|Known local NULL|GT not detector-matched|
|---|---:|---:|---:|---:|---:|---:|---:|
|4|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|
|5|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|
|6|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|
|7|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|
|8|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|
|9|10|171|186|78|2044|0|40|
|12|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|
|13|11|27|72|1104|9445|2|298|
|17|17|41|65|977|9236|2|724|
|20|102|148|321|3643|23746|7|2642|
|21|2|6|72|1333|7117|1|1070|
|24|420|672|1646|20077|69993|0|9538|

These columns are not mutually exclusive. Threshold rejection is an actual solver observation; NULL dominance and multiplicity-boundary evidence are separate overlapping flags, not invented exclusive causes. No pure candidate does not identify a causal detector failure. GT not detector-matched is a separate one-to-one detector coverage statistic and cannot be added to query errors.

|518 patch centers|Queries|Known wrong/missed|Correctable /known wrong|GTA Recall1 /pure-positive-full|VGGT ROI Recall1|DINO Recall1|
|---|---:|---:|---|---|---|---|
|0|56734|537|508 /537|30858 /35240|NOT_MEASURED|NOT_MEASURED|
|1-3|60286|347|327 /347|38999 /43423|NOT_MEASURED|NOT_MEASURED|
|4-15|19636|199|199 /199|16342 /17121|NOT_MEASURED|NOT_MEASURED|
|16+|1396|31|31 /31|1348 /1352|NOT_MEASURED|NOT_MEASURED|

Completed fit sequences contain 1200 Top8-reachable known ranking or actual primary wrong/missed queries; 1200 lie outside fixed V2. Reachable pure-known Top1 errors: 476. These overlap actual solver errors and can repeat along one identity. 451 adjacent-group pairs occur within the same scene/camera/identity error history; do not equate queries with independent error episodes.

DEV5 retrospective is saved separately. It cannot determine fit selection or AG/QD architecture. Scene1 saved actual solver evidence cannot establish candidate coverage in the other four scenes, where those fields remain NOT_MEASURED.
