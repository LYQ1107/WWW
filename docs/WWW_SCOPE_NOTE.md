# WWW scope and deadlines

[Official Research Track CFP](https://www2027.thewebconf.org/research-track-papers/) verified 2026-10-15: abstract 2026-10-18 AoE, full paper 2026-10-25 AoE. These end at 20:00 Asia/Shanghai on the following calendar day. Target track: Web Mining, Multimedia and Multilingual Content Analysis.

A submission must explain its Web scientific problem and track fit on the first page; using a Web dataset or API alone is insufficient. [Source](https://www2027.thewebconf.org/research-track-papers/)

The executable task queries a current detector target against clips from other cameras. Actual image reads, frozen causal GMT, feature publication, bounded index maintenance and query responses run online in sequence; labels enter post-run evaluation only. Separate scenes interleave by replay round and retain independent tracker banks/RNG. Frozen encoder weights are already shared serially across scene extractors; no cross-scene tracking/geometry state is shared.

Completed production tracker modes in this table: dinov2_dense_head, gmt_dense_head, gmt_off, native_fixed_prior, vggt_dense. Original experiments keep tracking at causal GMT OFF and vary retrieval features/scoring. Optional fixed-prior and dense B7 tracking-to-retrieval controls are separately gated; only completed metric artifacts enter this table. Local replay does not establish Internet-origin streams, live Web deployment, real user-query traffic or an independent Web scientific contribution.

Production retrieval results below use scene18 queries at every stream count. Exact query image/camera/frame/box cohorts and complete scene18 detector outputs are checked against the two-camera native reference. Recall@1/MRR are conditional on matched queries with positive gallery clips; coverage and all-request success remain in the metric artifacts.

| Method | Tracking | Cameras | Queries | Eligible | Recall@1 | MRR | Exact cohort/detections |
|---|---|---:|---:|---:|---:|---:|---|
| retrieval_dinov2_dense_head_tracking_2streams_full_v1 | dinov2_dense_head | 2 | 598 | 309 | 98.0583% | 0.990291 | True/True |
| retrieval_dinov2_dense_head_tracking_4streams_full_v1 | dinov2_dense_head | 4 | 598 | 309 | 98.0583% | 0.990291 | True/True |
| retrieval_gmt_10streams_full_v1 | gmt_off | 10 | 598 | 309 | 84.1424% | 0.885244 | True/True |
| retrieval_gmt_4streams_full_v1 | gmt_off | 4 | 598 | 309 | 97.7346% | 0.987864 | True/True |
| retrieval_gmt_dense_head_tracking_2streams_full_v1 | gmt_dense_head | 2 | 598 | 309 | 97.7346% | 0.987864 | True/True |
| retrieval_gmt_dense_head_tracking_4streams_full_v1 | gmt_dense_head | 4 | 598 | 309 | 97.7346% | 0.987864 | True/True |
| retrieval_gmt_v18_full_v1 | gmt_off | 2 | 598 | 309 | 97.7346% | 0.987864 | True/True |
| retrieval_native_prior_10streams_full_v1 | native_fixed_prior | 10 | 598 | 309 | 84.1424% | 0.885347 | True/True |
| retrieval_native_prior_2streams_full_v1 | native_fixed_prior | 2 | 598 | 309 | 97.7346% | 0.988673 | True/True |
| retrieval_native_prior_4streams_full_v1 | native_fixed_prior | 4 | 598 | 309 | 97.7346% | 0.988673 | True/True |
| retrieval_vggt_dense_tracking_2streams_full_v1 | vggt_dense | 2 | 598 | 309 | 98.0583% | 0.989752 | True/True |
| retrieval_vggt_dense_tracking_4streams_full_v1 | vggt_dense | 4 | 598 | 309 | 98.0583% | 0.989752 | True/True |
| retrieval_vggt_k10_10streams_full_v1 | gmt_off | 10 | 598 | 309 | 70.5502% | 0.769991 | True/True |
| retrieval_vggt_k10_4streams_full_v1 | gmt_off | 4 | 598 | 309 | 87.0550% | 0.891011 | True/True |
| retrieval_vggt_k10_v18_full_v1 | gmt_off | 2 | 598 | 309 | 90.2913% | 0.924582 | True/True |
| retrieval_vggt_k20_v18_full_v1 | gmt_off | 2 | 598 | 309 | 90.2913% | 0.925637 | True/True |
| retrieval_vggt_k5_v18_full_v1 | gmt_off | 2 | 598 | 309 | 89.9676% | 0.922593 | True/True |

The completed dense-B7 two-camera tracker feeding the native ranker has 1 Recall1 query corrections and 0 harms on 309 common eligible scene18 queries, with eligible gained/lost 0/0. Same detector query cohort and GT identities are independently verified. This is a small primary-head replay signal; completed matched-head controls are reported individually below, with declared-comparison status in the query audit. See GEO_STREAM_RETRIEVAL_QUERY_AUDIT.md; no unique VGGT or stable speed claim follows.

Completed matched publication control GMT-head2, fixed scene18: 598 exact detector queries/309 common eligible; R1 corrections/harms 0/0, eligibility gained/lost 0/0, MRR delta +0.000000, first-positive rank or eligibility changes 0. This is one primary-head local replay, not a completed stability or unique-foundation attribution result.

Completed matched publication control DINO-head2, fixed scene18: 598 exact detector queries/309 common eligible; R1 corrections/harms 1/0, eligibility gained/lost 0/0, MRR delta +0.002427, first-positive rank or eligibility changes 2. This is one primary-head local replay, not a completed stability or unique-foundation attribution result.

The complete four-camera dense-B7 replay, fixed scene1: 228 exact detector queries/185 common eligible, R1 corrections/harms 0/0, eligibility gained/lost 0/0, MRR delta +0.000893. Scene18 shares the two-camera corrected query; it is not an independent query replication. Per-scene cohorts are kept separate.

Completed matched publication control GMT-head4, fixed scene1: 228 exact detector queries/185 common eligible; R1 corrections/harms 0/0, eligibility gained/lost 0/0, MRR delta -0.000016, first-positive rank or eligibility changes 1. This is one primary-head local replay, not a completed stability or unique-foundation attribution result.

Completed matched publication control DINO-head4, fixed scene1: 228 exact detector queries/185 common eligible; R1 corrections/harms 1/1, eligibility gained/lost 0/0, MRR delta -0.000976, first-positive rank or eligibility changes 3. This is one primary-head local replay, not a completed stability or unique-foundation attribution result.

The complete four-camera dense-B7 replay, fixed scene18: 598 exact detector queries/309 common eligible, R1 corrections/harms 1/0, eligibility gained/lost 0/0, MRR delta +0.001888. Scene18 shares the two-camera corrected query; it is not an independent query replication. Per-scene cohorts are kept separate.

Completed matched publication control GMT-head4, fixed scene18: 598 exact detector queries/309 common eligible; R1 corrections/harms 0/0, eligibility gained/lost 0/0, MRR delta +0.000000, first-positive rank or eligibility changes 0. This is one primary-head local replay, not a completed stability or unique-foundation attribution result.

Completed matched publication control DINO-head4, fixed scene18: 598 exact detector queries/309 common eligible; R1 corrections/harms 1/0, eligibility gained/lost 0/0, MRR delta +0.002427, first-positive rank or eligibility changes 2. This is one primary-head local replay, not a completed stability or unique-foundation attribution result.

Same ordered-query audit of VGGT2 versus GMT-head2, scene18: VGGT/control R1 fixes 1/0, shared fixes 0, VGGT fixes not reproduced by this control 1. Equal aggregate gains are insufficient to infer the same corrected query; different corrections in a single replay do not establish stable unique VGGT benefit.

Same ordered-query audit of VGGT2 versus DINO-head2, scene18: VGGT/control R1 fixes 1/1, shared fixes 0, VGGT fixes not reproduced by this control 1. Equal aggregate gains are insufficient to infer the same corrected query; different corrections in a single replay do not establish stable unique VGGT benefit.

Same ordered-query audit of VGGT4 versus GMT-head4, scene18: VGGT/control R1 fixes 1/0, shared fixes 0, VGGT fixes not reproduced by this control 1. Equal aggregate gains are insufficient to infer the same corrected query; different corrections in a single replay do not establish stable unique VGGT benefit.

Same ordered-query audit of VGGT4 versus DINO-head4, scene18: VGGT/control R1 fixes 1/1, shared fixes 0, VGGT fixes not reproduced by this control 1. Equal aggregate gains are insufficient to infer the same corrected query; different corrections in a single replay do not establish stable unique VGGT benefit.


Same-gallery diagnostics retain the full native positive-gallery denominator. The sampled-observation native scorer is stronger than transferred association-trained projection/cosine scores in the completed scene18 tests; positive candidate coverage is 100%. These diagnose scoring/representation transfer and do not show all raw foundation features are useless. Their five-ranker producer timings are diagnostic cost, not production throughput. See retrieval_scoring_diagnostics.csv and the source metrics.

Query latency measures local scoring/response serialization after representations are available. Full replay-round timing separately charges reads, tracking, optional encoder work, publication, queries and log I/O. It is not HTTP/network or camera-capture-to-output latency. Source acquisition timing and a deadline-paced live camera service have not been verified. Index payload caps do not bound original GMT history or total process/GPU memory.

Remaining scope evidence: consolidate the measured stream/keyframe matrix and matched tracking-to-retrieval comparisons with the remaining stability/system evidence, preserve fixed-query comparisons, and establish an actual Web scientific problem/application. Current evidence does not justify PAPER_READY; this fit risk does not stop the technical research.
