# CV-PIL V2 read-only handoff and actor contract

The existing frozen 1000/300 TrackVGGT cohort is the only authorized current training source. Its SHA is in `manifests/cv_pil_v2_fixed_protocol.json`. The adapter verifies both source manifest and event payload against independently frozen cohort hashes, original GMT checkpoint/config hashes, selected native Top8 ID/index/score alignment, native history ownership and strictly past timing. It uses full 1152-D native re-ID features as an opaque vector, with no arbitrary VFCE/RPCE slice. Producer source SHA is retained.

`INPUT_ONLY` contains only physical query keys, scene/frame/camera, current frozen detector feature, original candidate Global IDs/scores and strictly past native histories. `inference_inputs` rejects additional fields and labeled history records. True person IDs, ambiguous ownership, positive/negative masks, primary outcomes and all GT-derived eligibility live exclusively in `OFFLINE_ONLY`. Alias tracks can be distinct legal Global IDs of the same true person: use multi-positive supervision and count success by scene/person. UNKNOWN remains unresolved, not a negative identity or a certified correction. Missing positive candidates are not invented. Same-time history is excluded; future history fails closed.

Legacy payloads may omit `native_top1_state`. The adapter reconstructs this offline convenience label from the existing SHA-bound query/positive/ambiguous labels and checks it against the frozen cohort pointer. No producer file is modified. The source labels certify **primary-before-memory** outcomes. They do not certify the final Bank/birth identity, or what a changed controller would commit.

The independent miner owns collection, selection and publication. CV-PIL will not rerun it or consume a directory merely because partial scene outputs exist. The following is a proposed consumer release header supported by `verify_mined_release`; an actual producer release must explicitly bind its own payload schema, source code, GT derivation, full solver outcome evidence and approved splits before downstream training. The header alone is not a correctness certificate:

```json
{
  "schema": "cvpil_native_release/v1",
  "status": "FROZEN_READY_FOR_CONSUMER",
  "feature_contract": "full_native_reid_1152_opaque_no_slicing",
  "checkpoint_sha256": "<original frozen Stage2 SHA256>",
  "config_sha256": "<original frozen config SHA256>",
  "gt_in_actor_inputs": false,
  "official_test_used": false,
  "development_selection_uses_gt": false,
  "fit_scene_ids": [4, 5, 6, 7, 8],
  "development_scene_ids": [12],
  "source_artifacts": {"<absolute authorized artifact path>": "<SHA256>"}
}
```

The consumer requires the release header's externally supplied SHA and a separately approved scene allowlist; producer scene declarations do not authorize new data. Release artifacts must resolve inside the agreed read-only runtime and match SHA. New native payload layouts require tested mapping and actual committed-error certification; no guessing of semantics or automatic integration. Until that handoff exists, P2 remains `WAITING_EXTERNAL_FROZEN_MINER_RELEASE`.

Subsequent actual-format handoff: the complete local producer inventory and per-scene manifests were accepted by their sealed SHA, frozen split, full400 prefix parity, actual candidate and commit checks. The proposed header above is not a mandatory producer format and Git synchronization is not a data-validity gate. See [the concrete source mapping and feature gap](CV_PIL_V2_SEALED_HANDOFF_AND_ASSET_GAP_ZH.md). P2 now requires missing native development feature/score/history assets rather than another metadata release. Recovery was post hoc engineering with previously exposed labels; its GT-free implementation does not make it a new blind evaluation cohort.

P0 completed: 12 adversarial tests and all six real source bundles passed. The 300-query development cohort has 18 known Top1 errors but zero labeled wrong/missed primary assignments. It has only 11 queries with a certified strictly past opposite-camera positive. All three preregistered 300-step CPU controls subsequently ran, with 19 total adapter/model/metric tests. See [the real result report](CV_PIL_V2_REAL_RESULTS_20261015_ZH.md); these opportunity limits prohibit claiming new native tracking success from retrieval metrics.
