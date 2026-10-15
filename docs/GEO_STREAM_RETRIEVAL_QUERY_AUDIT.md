# Paired detector-query retrieval outcomes

Each completed row verifies the exact physical query cohort and matched GT identities. Conditional Recall alone can hide corrections and harms or changed eligibility. Outcomes are recomputed from the retained original ranked responses after inference; old metric files are preserved. The recognized early four-counter schema is allowed only with explicit verification of every retained value.

| Method | Query scene | Requests | Common eligible | R1 fixes | R1 harms | Eligible gained/lost | Native R1 | Method R1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior2 | 18 | 598 | 309 | 0 | 0 | 0/0 | 0.977346 | 0.977346 |
| VGGT2 | 18 | 598 | 309 | 1 | 0 | 0/0 | 0.977346 | 0.980583 |
| Prior4 | 1 | 228 | 185 | 1 | 1 | 0/0 | 0.978378 | 0.978378 |
| VGGT4 | 1 | 228 | 185 | 0 | 0 | 0/0 | 0.978378 | 0.978378 |
| Prior4 | 18 | 598 | 309 | 0 | 0 | 0/0 | 0.977346 | 0.977346 |
| VGGT4 | 18 | 598 | 309 | 1 | 0 | 0/0 | 0.977346 | 0.980583 |

Pending comparison artifacts: 6. Their missing values are not zero. Full examples, cohort/metric hashes and MRR changes are in results/retrieval_paired_query_outcomes_v1.json.

These are query-rank changes, distinct from all-time tracking identity pairs or internal Hungarian decisions. A single primary-head local replay cannot establish stable VGGT-specific benefit, a speed advantage or WWW deployment.
