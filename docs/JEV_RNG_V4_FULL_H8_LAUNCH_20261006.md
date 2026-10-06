# Full H=8 launch record — 2026-10-06

The formal same-code video-07 single-vs-three-chunk gate passed before the
canonical build was launched. It compared all `3337/3337` records with zero
missing keys, zero record mismatches, exact canonical SHA, exact RNG
provenance, and all final-gate fields passing.

The authorized production partition is the locked 24-video partition with
`1,112,173` decisions. At launch, the scheduler started the largest remaining
videos on GPUs 4, 8, and 9. GPU 6 was reserved by the still-running corrected
video builder, GPU 1 had an unrelated process, GPU 0 remained deferred, and
foreign GPUs 2/3/5/7 were never used.

The build is still `RUNNING`; this document is an authorization/launch record,
not a final dataset or tracking result. After all shard manifests pass, the
supervisor runs finalization, compact conversion, the fixed TRAIN-only split,
and one `seed=20261003` three-way training round, then publishes a SHA-bound
summary to GitHub.
