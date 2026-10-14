# Geometry source audit — 2026-10-14

This is a separate VGGT+GMT project. No JEV policy, learned weights, perception cache, assignment changes or replay implementation is imported. Historical audit documents are read only to recover evaluator correctness and known exposure.

## GMT

[Official source](https://github.com/FoxCanned/GMT), local initial source `8fa96705`, missing data modules restored exactly from `dfa9ca8e`, mapper class declaration fix from `cbae5f9a`. The architecture files equal the initial source. `gtr/modeling/roi_heads/gtr_roi_heads.py::_forward_asso` and `_forward_asso_gmt` pool detector feature maps and attach `reid_features`; `_forward_transformer` consumes Instances and concatenated ReID, returns candidate logits, boxes, times and query indices. `association_head.py` produces target embeddings; `transformer.py` includes RPCE trajectory embeddings and randomly assigns embedding indices even in evaluation. Reset Python RNG before paired runs. Do not consume its RNG in a new module.

`gtr/modeling/meta_arch/gtr_rcnn.py::get_asso` selects current query against historical proposals. `_activate_asso` appends a zero unmatched logit and softmaxes. `run_first_tracker_plus` and `run_global_tracker_plus` aggregate proposal probabilities by track IDs with matrix multiplication before Hungarian. Track scores are sums, not calibrated track probabilities. A bounded residual may be inserted into proposal logits before activation, gated to different cameras; never add raw cosine to GTA sums. Memory uses averaged historical ReID and its own threshold; preserve this rule and use neutral evidence for unsupported memory fields.

`sliding_inference_GMT` reads view blocks, produces frame-major Instances, then full-video `_remove_short_track` and optional `_delay_cls`. These final operations are noncausal. Initial forward scans full input filenames; mapper loads complete videos. Offline and causal protocols require separate baselines. `gtr/evaluation/mot_evaluation.py` now aligns labels to frame-major output. Corrected labels are mandatory across protocols. Test YAML remains byte-identical (`bdaeca71...`). GMT README states Apache 2.0; dependent CenterNet2/Detectron2 notices remain intact.

## VGGT

[Source](https://github.com/facebookresearch/vggt), pinned `a288dd0f14786c93483e45524328726ab7b1b4ce`. `vggt/models/aggregator.py` returns a length-24 list with None at uncached indices; defaults (4,11,17,23), plus final layer. Each retained layer concatenates frame/global attention channels, and `patch_start_idx` excludes camera/register tokens. `vggt/models/vggt.py` has camera, depth, point and track heads. Camera extrinsics use camera-from-world OpenCV conventions. First-view learned camera tokens differ from subsequent views, so permutation consistency is measured, never assumed. `vggt/utils/load_fn.py` pad mode RGB [0,1], bicubic resize, rounds the short side to a multiple of 14, pads white to 518 square. Aggregator applies normalization internally. Our PadMapping follows the exact rounded x/y scale and offsets. No dense token persistence. Embedded DINO is a separately labeled diagnostic, not an independent pretrained DINOv2 control. Local isolated venv inherits torch 2.0 runtime read-only; added HF/einops dependencies do not change GMT environment. V100 requires FP16, not native BF16.

VGGT code uses Meta's VGGT custom license (`LICENSE.txt`, July 29 2025). Original `facebook/VGGT-1B` weights retain noncommercial terms; commercial weights have a separate gated release. This research uses the original public research weights; do not describe all weights as commercial licensed.

## VGGT-Det

[Source](https://github.com/yangcaoai/VGGT-Det-CVPR2026), `73ea4cf94a05fa50279ee0b6095fb06f6d042dac`. `projects/VGGTDet/vggtdet/vggtdet.py` freezes its VGGT encoder, optionally projects several levels and uses learned/point-based queries plus transformer cross-attention to combine views. Borrow only the hypothesis of target-conditioned multilevel representations. No MMDetection3D, PointNet CUDA, 3D box decoder or GT-driven query is copied. MIT project license; underlying VGGT has separate terms. Novelty risk: multilevel frozen VGGT aggregation alone is already established; tracking identity and complementary errors must be demonstrated.

## StreamVGGT

[Source](https://github.com/wzzheng/StreamVGGT), `05682bcb05683c4c71522e9cb618fc8ca349063c`. `src/streamvggt/models/aggregator.py` and `streamvggt.py::inference` use layer KV history and camera-head caches, single incoming frame in cached mode. This is a different trained streaming architecture, not a lossless switch for a frozen batch VGGT. A synchronized two-view group is not automatically equivalent to two single-view streaming steps. CC BY-NC-SA 4.0, with separate CroCo notices. Reference only at this stage; own bounded asynchronous worker must report when results actually finish.

## FastVGGT

[Source](https://github.com/mystorm16/FastVGGT), `6526e275a29572653a034762bb3c6c9ce280ff55`. `vggt/layers/attention.py` uses merge/unmerge from `merging/merge.py`, configurable merge ratios and global layers. This can change feature representations. Two synchronous images require measured accuracy and runtime; long-sequence headline speedups are not transferable. VGGT custom license in LICENSE.txt. Reference only until baseline feature inference runs; no speed multiplier claimed.

## Prior calibration-free tracking

[Paper](https://arxiv.org/abs/2607.22731), submitted 2026-07-22. It combines YOLOX/BoT-SORT/OsNet/HRNet with VGGT, pose-guided head lifting and hierarchical clustering with appearance/geometry and velocity gates on indoor cameras. Simple VGGT geometry + ReID matching is already prior art. Our proposed distinction is uncertain dynamic geometry for moving cameras, evidence of identity information beyond DINO, and identity-dependent online compute allocation. All three are hypotheses, with no positive result yet. No source or weights copied from this paper. Verify static versus dynamic assumptions and full evaluation details before final novelty claims.
