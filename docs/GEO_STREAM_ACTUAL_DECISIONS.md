# Actual primary-solver decisions — complete scene1

actual Hungarian solver calls and strict threshold before bank fallback/new-ID allocation; candidate truth uses only exact current solver reference proposals, one-to-one IoU>=.5 labels after both prediction gates.

known-pure reference tracks only; mixed/unmatched references and unmatched queries stay unknown; no-match correctness is conditional on recorded candidates, not global candidate recall; averaged-descriptor memory decisions excluded; no claim of full tracking correctness or all strict scenes from this scene1 diagnostic; separate runs have their own histories.

Both complete prediction/configuration gates and exact physical proposal/common GT-query cohorts pass. Detector proposals: 4527; common primary queries: 4524; baseline-only/method-only: 0/0. Conditional known fixes/harms: 32/1; unknown on either side: 386. These are pre-memory primary decisions, not final tracking correctness.

| Primary state | Baseline | Joint B7 |
|---|---:|---:|
| ambiguous_reference | 145 | 225 |
| correct_link | 4072 | 4028 |
| missed_known_positive | 149 | 112 |
| unknown_candidate_coverage | 4 | 5 |
| unknown_query | 154 | 154 |

| Transition on common query | Count |
|---|---:|
| ambiguous_reference -> ambiguous_reference | 143 |
| ambiguous_reference -> missed_known_positive | 2 |
| correct_link -> ambiguous_reference | 75 |
| correct_link -> correct_link | 3996 |
| correct_link -> missed_known_positive | 1 |
| missed_known_positive -> ambiguous_reference | 6 |
| missed_known_positive -> correct_link | 32 |
| missed_known_positive -> missed_known_positive | 109 |
| missed_known_positive -> unknown_candidate_coverage | 2 |
| unknown_candidate_coverage -> ambiguous_reference | 1 |
| unknown_candidate_coverage -> unknown_candidate_coverage | 3 |
| unknown_query -> unknown_query | 154 |

Averaged-descriptor memory solver calls excluded from GT correctness: 157/134. Unknown/mixed cases remain visible; fewer missed primary links alone does not prove overall tracking improvement. All raw hashes and the full scope are in actual_decision_full_scene1_cohort_verified_v2.json.
