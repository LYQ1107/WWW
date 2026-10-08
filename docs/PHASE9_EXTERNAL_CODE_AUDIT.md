# Phase IX external function-level code audit

All 11 repositories were read at fixed commits before changing tracker code. Only selected source/license/API metadata was downloaded; no external training environment installed and no upstream code copied. A live HEAD was resolved once where the request specified no hash. Raw audited source remains under `/home/liuyeqiang/WWW_jev_phase9_external_audit/20261008`.

## BoT-SORT

Commit: `251985436d6712aaf682aaaf5f71edb4987224bd`; root license: MIT (root); bundled dependencies have separate licenses.

STrack.update_features normalizes current feature and EMA smoothed feature (alpha=.9). BoTSORT.update first combines tracked/lost pools with high-confidence detections, motion compensation and gated appearance/IoU minimum; second associates unmatched tracked tracks with actual low-confidence detections using IoU. matching.embedding_distance uses cosine, fuse_motion gates Mahalanobis then blends, fuse_score multiplies IoU similarity by confidence.

Baseline engineering may improve association without JEV. Changing perception, Kalman state or memory EMA here would confound frozen-MATCH comparisons; only separately identified score controls fit this experiment.

- [tracker/bot_sort.py](https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/bot_sort.py) — SHA256 `ba6dd32ad4b96ee49d6f70654159269bd45ce15fd0fe5007f3a1b81bc257edf0`
- [tracker/matching.py](https://github.com/NirAharon/BoT-SORT/blob/251985436d6712aaf682aaaf5f71edb4987224bd/tracker/matching.py) — SHA256 `020b2215c20467b31f7d8a977a2f557b8d1838f8baa5a5be54c0b8d14bbee4e1`

**WHAT DID WE LEARN?** Baseline engineering may improve association without JEV. Changing perception, Kalman state or memory EMA here would confound frozen-MATCH comparisons; only separately identified score controls fit this experiment.

## CAMELTrack

Commit: `46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2`; root license: Apache-2.0.

GAFFE.forward concatenates detection, tracklet and always-valid cls tokens; inverts valid masks for Transformer padding. TemporalEncoder.forward aggregates past feature sequences with age embeddings and cls token, separately from group interactions. CAMEL.forward tokenizes, merges temporally, applies GAFFE, then similarity; compute_loss uses identity-supervised embedding loss, not future causal value. CAMELDataset.__getitem__ selects features with image_id<=current image, creates per-video global training IDs and age; pad_dict/collate_fn preserve missing targets. OcclusionSampler.sample_generator weights frames by overlap counts; GapSampler by no-observation counts. hungarian_algorithm masks both sets, solves whole scores then checks threshold.

Context is learned over all current detections/tracklets and past temporal evidence. Borrow difficult-stratum design and empty-set safety, not its GT/gallery samplers verbatim. Our proposed difference is controlled actual-state intervention supervision of delayed externalities; no measured architectural advantage yet.

- [cameltrack/architecture/gaffe.py](https://github.com/TrackingLaboratory/CAMELTrack/blob/46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2/cameltrack/architecture/gaffe.py) — SHA256 `fad8bef556ad55eb47515026074297508ed495cbeedcdea48d5bed757b12cc26`
- [cameltrack/architecture/temporal_encoder.py](https://github.com/TrackingLaboratory/CAMELTrack/blob/46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2/cameltrack/architecture/temporal_encoder.py) — SHA256 `c28a9c9abcead128e2c45003dcc836bf0aa001d215c4bb8bcb54c8555c996293`
- [cameltrack/camel.py](https://github.com/TrackingLaboratory/CAMELTrack/blob/46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2/cameltrack/camel.py) — SHA256 `d2a6139ac93f057fe8b6909d103f2d363ff7e3c67c853846a5d9d2c15dd7fed2`
- [cameltrack/train/dataset.py](https://github.com/TrackingLaboratory/CAMELTrack/blob/46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2/cameltrack/train/dataset.py) — SHA256 `ea5cae7fe66fe6739fa5e4e8a8e1c05dbc72a126741514e9849f6a42f7fcc034`
- [cameltrack/train/sampler.py](https://github.com/TrackingLaboratory/CAMELTrack/blob/46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2/cameltrack/train/sampler.py) — SHA256 `ad0fcdd89228a148f01a2956a8c267c9e56ac432e796ba5bbd48a41719cb9d36`
- [cameltrack/utils/assignment_strats.py](https://github.com/TrackingLaboratory/CAMELTrack/blob/46a74bb22a28d2d699b4c5c5e317a26d3b87f1e2/cameltrack/utils/assignment_strats.py) — SHA256 `7772c3db7c05273cfffd75711a1d916c0733d87f72d70031c0e01f0ff72dfcf0`

**WHAT DID WE LEARN?** Context is learned over all current detections/tracklets and past temporal evidence. Borrow difficult-stratum design and empty-set safety, not its GT/gallery samplers verbatim. Our proposed difference is controlled actual-state intervention supervision of delayed externalities; no measured architectural advantage yet.

## Deep-OC-SORT

Commit: `6bb51d027b137233f5c520b6fcc4f2ae387a6ba9`; root license: MIT (root); bundled dependencies have separate licenses.

OCSort.update computes confidence-dependent embedding alpha, predicts tracks, calls appearance/motion association, then rematches residual detections to last observed boxes (OCR). Successful matches update track embeddings with detection-dependent EMA; unmatched tracks receive update(None), unmatched detections create tracks.

Past observation recovery and detection-confidence update are meaningful ordinary controls. This is not future-rollout supervision and cannot change GMT memory in the controlled MATCH architecture comparison.

- [trackers/integrated_ocsort_embedding/ocsort.py](https://github.com/GerardMaggiolino/Deep-OC-SORT/blob/6bb51d027b137233f5c520b6fcc4f2ae387a6ba9/trackers/integrated_ocsort_embedding/ocsort.py) — SHA256 `2f4d4ccf8ce7f703367607bcb45d7a6b92ea496500881bf3599fe0a65563388c`

**WHAT DID WE LEARN?** Past observation recovery and detection-confidence update are meaningful ordinary controls. This is not future-rollout supervision and cannot change GMT memory in the controlled MATCH architecture comparison.

## MOTIP

Commit: `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`; root license: Apache-2.0.

IDDecoder.forward appends ID-word embeddings to trajectory features and an empty word to unknown detections. Its cross-attention mask blocks history time>=query time; _forward_a_layer includes same-time unknown self-attention after first layer and temporal-relative cross-attention. TrajectoryModeling.forward adapts features with FFN/residual/norm. RuntimeTracker._get_id_pred_labels constructs ID context; _hungarian_assignment repeats NEW columns, resolves globally, applies threshold and tracked-ID availability; _assign_newborn_id_labels recycles an ID vocabulary while mapping to runtime IDs. IDDecoder.shuffle permutes vocabulary weights.

Dynamic candidate scoring is related to in-context ID decoding. Runtime IDs must remain references, not learned integer labels. MOTIP uses a contextual ID vocabulary; proposed JEV scores actual available references with correctness/causal utility heads. Neither this distinction nor candidate attention proves superiority. runtime_tracker.py uses Python3.10 match/case, so Python3.9 AST/import failure is interpreter compatibility, not upstream source corruption.

- [models/motip/id_decoder.py](https://github.com/MCG-NJU/MOTIP/blob/ffc0e905ac196a603027eca8d18fb0dff48c8bcc/models/motip/id_decoder.py) — SHA256 `a16cf34bb16fa14a17d715231af1439cb3f1c1fa15d18341e2a84c61cdf59d81`
- [models/motip/trajectory_modeling.py](https://github.com/MCG-NJU/MOTIP/blob/ffc0e905ac196a603027eca8d18fb0dff48c8bcc/models/motip/trajectory_modeling.py) — SHA256 `0bdf92640347ec2b719ac56082aec307eeb6c3fd39c1cc87ea26e197eca58166`
- [models/runtime_tracker.py](https://github.com/MCG-NJU/MOTIP/blob/ffc0e905ac196a603027eca8d18fb0dff48c8bcc/models/runtime_tracker.py) — SHA256 `f0ac5da7c7331f8d11336924ad2c1996a15bf69fd99fe1b6de8bc573c96e8497`

**WHAT DID WE LEARN?** Dynamic candidate scoring is related to in-context ID decoding. Runtime IDs must remain references, not learned integer labels. MOTIP uses a contextual ID vocabulary; proposed JEV scores actual available references with correctness/causal utility heads. Neither this distinction nor candidate attention proves superiority. runtime_tracker.py uses Python3.10 match/case, so Python3.9 AST/import failure is interpreter compatibility, not upstream source corruption.

## MeMOTR

Commit: `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80`; root license: MIT.

QueryUpdater.update_tracks_embedding confidence-gates reference points/query writes, combines confidence-weighted current and last outputs as short memory, attends with long memory, then updates long memory by lambda only for positive confidence. MeMOTR.forward builds masked detection/track queries and reference points, applies transformer, emits class/box/query outputs consumed by update.

Reference for later WRITE supervision; memory-vector or score changes alone do not establish downstream identity utility. MATCH experiments preserve original GMT writes and stale-bank rules except explicit semantic NEW veto.

- [models/query_updater.py](https://github.com/MCG-NJU/MeMOTR/blob/eb7a177b9cbcb89742ec69b2545ab3af2ea31a80/models/query_updater.py) — SHA256 `112864688ef785a40d8837e678ada82d25d974ac8610c93f823ec91392c570b4`
- [models/memotr.py](https://github.com/MCG-NJU/MeMOTR/blob/eb7a177b9cbcb89742ec69b2545ab3af2ea31a80/models/memotr.py) — SHA256 `e87609757872ce1e04c26ea53f178f8e34cea999e4b3102312bc2b739f6f395d`

**WHAT DID WE LEARN?** Reference for later WRITE supervision; memory-vector or score changes alone do not establish downstream identity utility. MATCH experiments preserve original GMT writes and stale-bank rules except explicit semantic NEW veto.

## OC_SORT

Commit: `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`; root license: MIT.

association.associate computes observation-centric velocity/detection angle compatibility, masks unknown previous observations, weights angle cost by detection confidence and adds it to IoU. If threshold adjacency is unambiguous it directly returns edges, else linear_assignment; _filter_matches rejects IoU below threshold.

Explicit kinematic evidence helps recovery; these inputs require real available observation history. Do not assume interchangeable multi-camera geometry or import a new motion tracker into GMT without a separate confound control.

- [trackers/ocsort_tracker/association.py](https://github.com/noahcao/OC_SORT/blob/8462e7e729a93ccd3bd995c0a79a890336cb3a0b/trackers/ocsort_tracker/association.py) — SHA256 `c009a78eaa1c35c1d61217506255051ba882a7b99f9505cdec64da474ff0fbd7`

**WHAT DID WE LEARN?** Explicit kinematic evidence helps recovery; these inputs require real available observation history. Do not assume interchangeable multi-camera geometry or import a new motion tracker into GMT without a separate confound control.

## SUSHI

Commit: `ff1952b408835007f07d1fc78760872625fa6ae4`; root license: MIT.

MetaLayer.forward updates edges from endpoint nodes, then nodes from adjacent edges. TimeAwareNodeModel.forward pools separate incoming/outgoing flows. MOTMPNet.forward repeatedly attaches initial representations, message-passes and classifies edges. HICLTracker.hicl_forward predicts/projects edges, takes connected components, updates hierarchy and pooled features. track constructs graphs over start/end subsequences and merges overlapping subsequence identities.

Shared-edge competition motivates online conflict context. Its subsequence graph may see future frames; offline hierarchical inference/post-hoc linking cannot be presented as current online candidate inference. Our runtime must receive only prefix state/current detections.

- [src/models/mpntrack.py](https://github.com/dvl-tum/SUSHI/blob/ff1952b408835007f07d1fc78760872625fa6ae4/src/models/mpntrack.py) — SHA256 `dba2c133bd9a970a5fe4b3789d7a928edf8145d1f736173555c435b669a8c93f`
- [src/tracker/hicl_tracker.py](https://github.com/dvl-tum/SUSHI/blob/ff1952b408835007f07d1fc78760872625fa6ae4/src/tracker/hicl_tracker.py) — SHA256 `f13b2018b3fbdc14700619e97d3e1e37fdaa04d23e3c9f3689b931d9df613993`

**WHAT DID WE LEARN?** Shared-edge competition motivates online conflict context. Its subsequence graph may see future frames; offline hierarchical inference/post-hoc linking cannot be presented as current online candidate inference. Our runtime must receive only prefix state/current detections.

## StrongSORT

Commit: `ee995076da5083e28d0da1f885297df62705ebd7`; root license: GPL-3.0.

Tracker._match splits confirmed/unconfirmed tracks, calls appearance gated matching_cascade for confirmed, then min_cost_matching IoU for unconfirmed and recently unmatched. update marks misses/initiates tracks and fits appearance metric, with opt.EMA changing feature-buffer treatment.

Matching cascade is an online age-prioritized rule. Do not import AFLink/post-hoc future linking as online association. Source is read for design; no GPL code is copied into WWW.

- [deep_sort/tracker.py](https://github.com/dyhBUPT/StrongSORT/blob/ee995076da5083e28d0da1f885297df62705ebd7/deep_sort/tracker.py) — SHA256 `9fb02cefa5098c7d4351ae727c514718694fb55b7c8a0d8822982925b056c0ec`

**WHAT DID WE LEARN?** Matching cascade is an online age-prioritized rule. Do not import AFLink/post-hoc future linking as online association. Source is read for design; no GPL code is copied into WWW.

## TrackTrack

Commit: `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da`; root license: MIT.

Tracker.update builds high/low/deleted-high pools, applies motion correction/prediction, matches tracked/lost then new tracks, removes old tracks and initializes residual detections. iterative_assignment fuses IoU/cosine/confidence/angle with low/deleted penalties, repeatedly selects mutual minima with decreasing threshold and removes assigned rows/columns. track_aware_nms suppresses births overlapping active tracks or stronger detections; init_tracks applies it.

Useful rule controls for candidate competition and false births. Deleted detections are only available when a detector actually supplied them; frozen GMT cache cannot synthesize them. This iterative mutual-minimum algorithm is not Hungarian or JEV causal learning.

- [3. Tracker/trackers/tracker.py](https://github.com/kamkyu94/TrackTrack/blob/ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da/3.%20Tracker/trackers/tracker.py) — SHA256 `b3289f19060502d31dccebf88a2c98986afb61e1be8c3f13258726c34a725f62`
- [3. Tracker/trackers/utils.py](https://github.com/kamkyu94/TrackTrack/blob/ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da/3.%20Tracker/trackers/utils.py) — SHA256 `84d5daa45b4f0fb82bec3e89b8935bffc7f93f28b621270cccc9721163f87c79`

**WHAT DID WE LEARN?** Useful rule controls for candidate competition and false births. Deleted detections are only available when a detector actually supplied them; frozen GMT cache cannot synthesize them. This iterative mutual-minimum algorithm is not Hungarian or JEV causal learning.

## qdtrack

Commit: `c5b10472d7bdd3b9ab75255dd10e48e21f48c54f`; root license: Apache-2.0.

QuasiDenseEmbedTracker.match sorts detections by confidence, suppresses overlaps, computes (row-softmax+column-softmax)/2 over embedding affinity, optionally class-masks, greedily claims memo identities and starts sufficiently confident NEW tracks. update_memo maintains embedding momentum, velocity and frame-expiry, plus backdrop memory.

A frozen bidirectional-normalized score with shared Hungarian is an ordinary candidate-competition control, not a faithful whole-QDTrack reproduction. Its original sequential matching and perception differ. Test normalization alone before crediting JEV structure.

- [qdtrack/models/trackers/quasi_dense_embed_tracker.py](https://github.com/SysCV/qdtrack/blob/c5b10472d7bdd3b9ab75255dd10e48e21f48c54f/qdtrack/models/trackers/quasi_dense_embed_tracker.py) — SHA256 `a2fe5c209ee0d0c448d82207b41fcda4b78dba4a6ca865f9d50705b3061fcf6e`

**WHAT DID WE LEARN?** A frozen bidirectional-normalized score with shared Hungarian is an ordinary candidate-competition control, not a faithful whole-QDTrack reproduction. Its original sequential matching and perception differ. Test normalization alone before crediting JEV structure.

## set_transformer

Commit: `73432c640ac78140496d6738416c54d32c686d65`; root license: MIT.

MAB.forward projects Q/K/V and multihead scaled dot-product attention, residual and optional norms. SAB composes self-attention, ISAB uses inducing tokens, PMA pools with learned seeds. These signatures do not accept dynamic padding masks or provide legal NEW.

Permutation-compatible architecture is useful but must add legal masks, empty-set protection, per-row NEW and identity-reference preserving equivariance. The unmodified public implementation is not sufficient for variable legal candidate sets.

- [modules.py](https://github.com/juho-lee/set_transformer/blob/73432c640ac78140496d6738416c54d32c686d65/modules.py) — SHA256 `26c11c9f18209cd2ddcf52073a28b0052b356673ab356a1a5bf8ec41e9c95787`

**WHAT DID WE LEARN?** Permutation-compatible architecture is useful but must add legal masks, empty-set protection, per-row NEW and identity-reference preserving equivariance. The unmodified public implementation is not sufficient for variable legal candidate sets.
