# G1 full TRAIN spatial-support summary

The complete immutable TRAIN cache was audited across all 24 scenes (28,754 synchronized timestamps). Using GT only for post-hoc diagnostics:

- Matched observations: **543,504**.
- Same-person pairs: **180,256**.
- Same-person geometrically consistent: **141,495 (78.49%)**.
- Different-person candidate pairs: **5,873,192**.
- Different-person ambiguous pairs explicitly retained: **305,260**.
- Unmatched observations: **13,498**.

The scene spread is large (Scenes 12, 13, 18, 19, and 20 are materially weaker than Scenes 1, 4, 5, 8, 9, and 11). This is a TRAIN geometry diagnostic, not a claim of final identity or MOT improvement. Full per-scene CSV/JSON remains in the local/NAS experiment directory; the compact summary is versioned on the isolated branch.
