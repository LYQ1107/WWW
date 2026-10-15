# WWW scope and deadlines

[Official Research Track CFP](https://www2027.thewebconf.org/research-track-papers/) verified 2026-10-15: abstract 2026-10-18 AoE, full paper 2026-10-25 AoE. These end at 20:00 Asia/Shanghai on the following calendar day. Target track: Web Mining, Multimedia and Multilingual Content Analysis.

A submission must explain its Web scientific problem and track fit on the first page; using a Web dataset or API alone is insufficient. [Source](https://www2027.thewebconf.org/research-track-papers/)

The executable task queries a current detector target against clips from other cameras. Actual image reads, frozen causal GMT, feature publication, bounded index maintenance and query responses run online in sequence; labels enter post-run evaluation only. Separate scenes interleave by replay round and retain independent tracker banks/RNG. Frozen encoder weights are already shared serially across scene extractors; no cross-scene tracking/geometry state is shared.

Completed production tracker modes in this table: gmt_off, native_fixed_prior. Original experiments keep tracking at causal GMT OFF and vary retrieval features/scoring. Optional fixed-prior and dense B7 tracking-to-retrieval controls are separately gated; only completed metric artifacts enter this table. Local replay does not establish Internet-origin streams, live Web deployment, real user-query traffic or an independent Web scientific contribution.

Production retrieval results below use scene18 queries at every stream count. Exact query image/camera/frame/box cohorts and complete scene18 detector outputs are checked against the two-camera native reference. Recall@1/MRR are conditional on matched queries with positive gallery clips; coverage and all-request success remain in the metric artifacts.

| Method | Tracking | Cameras | Queries | Eligible | Recall@1 | MRR | Exact cohort/detections |
|---|---|---:|---:|---:|---:|---:|---|
| retrieval_gmt_10streams_full_v1 | gmt_off | 10 | 598 | 309 | 84.1424% | 0.885244 | True/True |
| retrieval_gmt_4streams_full_v1 | gmt_off | 4 | 598 | 309 | 97.7346% | 0.987864 | True/True |
| retrieval_gmt_v18_full_v1 | gmt_off | 2 | 598 | 309 | 97.7346% | 0.987864 | True/True |
| retrieval_native_prior_2streams_full_v1 | native_fixed_prior | 2 | 598 | 309 | 97.7346% | 0.988673 | True/True |
| retrieval_vggt_k10_10streams_full_v1 | gmt_off | 10 | 598 | 309 | 70.5502% | 0.769991 | True/True |
| retrieval_vggt_k10_4streams_full_v1 | gmt_off | 4 | 598 | 309 | 87.0550% | 0.891011 | True/True |
| retrieval_vggt_k10_v18_full_v1 | gmt_off | 2 | 598 | 309 | 90.2913% | 0.924582 | True/True |
| retrieval_vggt_k20_v18_full_v1 | gmt_off | 2 | 598 | 309 | 90.2913% | 0.925637 | True/True |
| retrieval_vggt_k5_v18_full_v1 | gmt_off | 2 | 598 | 309 | 89.9676% | 0.922593 | True/True |

Same-gallery diagnostics retain the full native positive-gallery denominator. The sampled-observation native scorer is stronger than transferred association-trained projection/cosine scores in the completed scene18 tests; positive candidate coverage is 100%. These diagnose scoring/representation transfer and do not show all raw foundation features are useless. Their five-ranker producer timings are diagnostic cost, not production throughput. See retrieval_scoring_diagnostics.csv and the source metrics.

Query latency measures local scoring/response serialization after representations are available. Full replay-round timing separately charges reads, tracking, optional encoder work, publication, queries and log I/O. It is not HTTP/network or camera-capture-to-output latency. Source acquisition timing and a deadline-paced live camera service have not been verified. Index payload caps do not bound original GMT history or total process/GPU memory.

Remaining scope evidence: complete the planned stream/keyframe matrix and candidate tracking-to-retrieval evaluation, preserve fixed-query comparisons, and establish an actual Web scientific problem/application rather than changing a tracking title. Current evidence does not justify PAPER_READY; this fit risk does not stop the technical research.
