# Phase VII publication scope

The user's 2026-10-08 clarification is to upload only necessary review artifacts. It supersedes interpreting "push code and experiment results" as uploading every raw prediction, journal and mutable-state snapshot.

Publish source changes, exact configurations/seeds/splits, required reports, aggregate and per-video metrics, executable schemas/loaders/tests, compact diagnostic examples, figures and provenance/SHA256 manifests. Preserve negative results and failed gates in these reports.

Keep complete predictions, per-frame decisions, large mechanism logs, fork snapshots, perception caches, videos and bulk generated labels in the verified local runtime archive. Do not force-add these files merely because an experiment finished. Provide selected evidence when an actual review needs it. A model checkpoint is an optional reproducibility artifact, not a default upload; record its hash and location first.

The full evidence batch had already reached GitHub when the smaller-publication request was checked. The research revision `a90604d636b5efea960e6594226da5b0f1a8f52e` was verified against the remote branch. Those published records and original Phase V/VI/B2 anchors are preserved. Ignoring new evidence prevents ordinary staging of later large archives; it does not remove already tracked files or shrink existing Git history.

Local full evidence: `/home/liuyeqiang/WWW_jev_phase7_runtime/20261008_p05_v1`. `PUBLICATION_INTEGRITY.json` records 102 archive manifests and 6,851 verified source/archive pairs, with 2,274,043,747 archive bytes. This complete archive supports forensic replication, while the [final report](PHASE7_FINAL_RESEARCH_REPORT.md) and summarized JSON results support normal scientific review.

The temporary transport branch was deleted after the research branch was verified. The flat content-addressed transport supplied durable remote batches, but Git's path-based traversal retransmitted those blobs in the final research push; it did not eliminate that transfer. Future compact publication avoids this transport approach.
