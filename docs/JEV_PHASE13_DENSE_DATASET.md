# Phase XIII dense association data

All 12/13/14/16 TRAIN and 17/18/19 development payloads use freshly computed, unjittered Stage1 VFCE1024. Real B1 commits produce the history; no GTA scores, teacher forcing, future labels, or corrective filtering. All actual MATCH and queried REACT groups remain present.

A detection is aligned to current TRAIN GT by one-to-one Hungarian IoU≥0.5. A historical predicted ID with multiple GT labels remains UNKNOWN. Potentially correct contaminated IDs do not authorize a DEFER label. Unknown options never become training negatives. Loss/calibration normalizers cover certified options only; full online inference still admits all legal candidates. START_NEW labels mean new to observed memory; scene-new counts are separate.

Version v1 is preserved. v2 changes uncertain labels and fixes the report field that mistakenly read `view_num` as `id_count`; every actor tensor, reference, and event ordering is unchanged. Per-video SHA, natural distributions, candidate support denominators, and frozen Tiny examples are in the linked JSON reports.

There are 91624 ordinary certified active positive rows in TRAIN, 7 stale recovery positives, and 24 new-to-memory positives. REACT is unqualified for formal learning; MEMORY has no WRITE/KEEP supervision. Neither head may mutate native state. Their frozen fallbacks are shared by every formal method.

Stage1 historical pretraining included all 24 TRAIN sequences. Development scenes are wood/park, TRAIN scenes path/football, but numeric IDs alone cannot certify person disjointness. No independent full-system test claim is possible with inherited Stage1. video20/21/22 remain sealed. Sample near-duplicate checks do not establish exhaustive absence.
