# Correctable identity error analysis

Status: INCOMPLETE. Fit totals exclude validation scene12.

|Scene|A ranking|B actual known failure|Threshold rejected (all)|No pure candidate|Known mixed history|Unknown reference observation|Known local NULL|GT not detector-matched|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|4|24|242|301|100|NOT_MEASURED|NOT_MEASURED|0|281|
|5|0|52|62|33|NOT_MEASURED|NOT_MEASURED|0|2|
|6|0|29|43|39|NOT_MEASURED|NOT_MEASURED|0|0|
|7|8|63|72|135|NOT_MEASURED|NOT_MEASURED|0|77|
|8|6|25|33|78|NOT_MEASURED|NOT_MEASURED|0|121|
|9|10|171|186|78|NOT_MEASURED|NOT_MEASURED|0|40|
|12|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|
|13|11|27|72|1104|NOT_MEASURED|NOT_MEASURED|2|298|
|17|17|41|65|977|NOT_MEASURED|NOT_MEASURED|2|724|
|20|102|148|321|3643|NOT_MEASURED|NOT_MEASURED|7|2642|
|21|2|6|72|1333|NOT_MEASURED|NOT_MEASURED|1|1070|
|24|420|672|1646|20077|NOT_MEASURED|NOT_MEASURED|0|9538|

These columns are not mutually exclusive. Threshold rejection is an actual solver observation; NULL dominance and multiplicity-boundary evidence are separate overlapping flags, not invented exclusive causes. No pure candidate does not identify a causal detector failure. GT not detector-matched is a separate one-to-one detector coverage statistic and cannot be added to query errors.

|518 patch centers|Queries|Known wrong/missed|Correctable /known wrong|GTA Recall1 /pure-positive-full|VGGT ROI Recall1|DINO Recall1|
|---|---:|---:|---|---|---|---|
|0|56860|540|511 /540|30969 /35354|NOT_MEASURED|NOT_MEASURED|
|1-3|62619|390|369 /390|41088 /45611|NOT_MEASURED|NOT_MEASURED|
|4-15|31376|372|371 /372|27126 /28546|NOT_MEASURED|NOT_MEASURED|
|16+|4056|225|225 /225|3806 /3942|NOT_MEASURED|NOT_MEASURED|

Completed fit sequences contain 1619 Top8-reachable known ranking or actual primary wrong/missed queries; 1489 lie outside fixed V2. Reachable pure-known Top1 errors: 514. These overlap actual solver errors and can repeat along one identity. 578 adjacent-group pairs occur within the same scene/camera/identity error history; do not equate queries with independent error episodes.

DEV5 retrospective is saved separately. It cannot determine fit selection or AG/QD architecture. Scene1 saved actual solver evidence cannot establish candidate coverage in the other four scenes, where those fields remain NOT_MEASURED.
