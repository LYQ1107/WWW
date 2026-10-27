WorldMOT V5 generated-input V100 readiness

The registered Membership Head passes8 generated-input CPU/CUDA checks in FP32. This is a supplementary hardware check. No VisionTrack image, annotation, perception cache or trained checkpoint is read. No optimizer is created, weights are neither updated nor saved, and the three selection workers retain their original process identities.

B2/B3/B4 each have606,468 parameters and share initialization tensor SHA-256 `3fb3f3d6336e8aa32d635ecff3c05929b0d4b8f265f4a0ccf56771814d13ed23`. The CUDA model's initialization hash exactly matches CPU. The frozen training/inference source remains `c4f9d1c0b2a290062aba52ae7c0d322176ae5376f9c85688f74d89159e16f733`.

| Generated history objects | Generated observations | CPU FP32 forward/backward | V100 FP32 forward/backward |
|---:|---:|---|---|
| 0 | 8 | passed | passed |
| 8 | 0 | passed | passed |
| 96 | 72 | passed | passed |
| 192 | 144 | passed | passed |

The checks verify finite outputs and gradients, `M×N` membership logits and `M×2` camera-NULL logits, and unchanged weights after backward-only execution. The device was a genuinely idle Tesla V100-DGXS-32GB, reserved under the shared launch lock and six-device limit. Its lease was released after actual process termination and receipt validation.

CUDA peak allocated memory was303,200,768 bytes for this head-only fixture. It excludes the detector, Omega, optimizer, complete world-object bank and full-video inputs. This is neither a complete training memory bound nor a model/full-pipeline FPS measurement. Full24 training and actual video timing remain required.

The executable is `worldmot/analysis/preflight_worldobject_v5_cuda.py`; the complete receipt and checker stdout are in `worldmot/results/v5_cuda_head_preflight.json`. The existing common12/16 TRAIN-only rule, final fresh-all24 eligibility and formal TEST gates are unchanged.
