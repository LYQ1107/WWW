# R2 supplementary fuser failure classification (2026-10-14)

This is a read-only audit of two preserved FAILED registry entries. It did not restart a process, rewrite a registry status, alter a checkpoint or label, or change a gate.

- Status: `COMPLETE_FAILURE_REVIEW_WITH_NO_RESTART`
- Queue terminal: `True`
- Live jobs: `[]`
- Automatic restart: `False`
- Preregistration stop rule: `Engineering failure retained; no automatic retraining or threshold/seed selection; all four registered final fusers are run irrespective of sign`

| Preserved failed entry | Classification | Corrected successor | Result boundary |
|---|---|---|---|
| `R2_FUSION_explicit_scores_identity_v2` | `INPUT_ARTIFACT_MISSING` | `R2_FUSION_explicit_scores_identity_v3` (`COMPLETE`) | failed attempt remains excluded from success claims |
| `R2_FUSION_explicit_scores_tiny_v1` | `INPUT_SCHEMA_MISMATCH` | `R2_FUSION_explicit_scores_tiny_v2` (`COMPLETE`) | failed attempt remains excluded from success claims |

The identity failure is a missing parent RESULT routing artifact. The Tiny failure is a schema contract mismatch (`anchor_visual`). Both failures occurred before a valid final comparison for that attempt; later v2/v3 entries completed the corresponding frozen diagnostics and are the artifacts used by the bounded fusion result index.

The corrected successors do not erase the earlier failures. The formal claim is therefore: the bounded R2/R3 result package is terminal and the four supplementary fuser diagnostics have usable completed artifacts, while the two earlier engineering attempts remain FAILED and are not evidence of scientific success or failure of the JEV hypothesis.

No automatic retry is authorized under the current frozen queue. A further rerun would require a new preregistered execution identity and an explicit input-contract repair.
