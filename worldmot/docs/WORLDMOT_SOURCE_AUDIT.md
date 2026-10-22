# Source audit

The official PanSt3R checkout is `/data1/liuyeqiang/PanSt3R_official_20261021` at commit `0253909812ce570d35f2325a20a591e4785f7d7d`. Its MUSt3R dependency is `/data1/liuyeqiang/MUSt3R_official_20261021` at commit `efc5b08fbbaba814706880932ba2497472096925`, with the official DUSt3R/CroCo submodules initialized.

The adapter follows the official interfaces:

- `PanSt3R.from_checkpoint()` constructs the checkpoint-declared MUSt3R encoder, decoder, DINO encoder, and PanopticDecoder.
- `forward_inference_multi_ar()` is called with exactly two keyframes, `use_retrieval=False`, and the class vocabulary `['person']`.
- `panoptic_inference_v2()` is used for standard query/mask postprocessing; QUBO is not used.
- `must3r.engine.inference.postprocess(..., compute_cam=True)` decodes pointmaps, confidence, focal, and camera-to-world estimates.
- `set_vocab()` is executed against the official SigLIP text model. A missing asset is a dependency failure, never silently replaced by a category-id rewrite.

The v2 checkpoint is the official download and was verified with MD5 `6a7a47a2ea635de05af45ded8dcabd57`. It is intentionally outside this Git repository.

SparseQID was audited separately at `/data1/liuyeqiang/SparseQID_official_20261021`, commit `7aea61d9882946f7284c8faa6ae2263af6495360`. Its `SparseQIDTracker` confirms useful lifecycle concerns (active labels, newborn slots, Hungarian/object-max assignment, miss tolerance, and memory windows). WorldMOT does not load its detector, checkpoint, or identity decoder; the initial WorldMOT memory is deterministic and uses world geometry plus current query features.
