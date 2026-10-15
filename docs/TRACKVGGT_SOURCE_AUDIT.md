# TrackVGGT source audit

Pinned inheritance861c8710; new branch research/trackvggt-identity-query-www27-20261015. Full prior reports read; old ROI/pose/sparsity experiments are retained as negative evidence rather than repeated as new ideas.

## Actual AG mechanism and adaptation

[VGGT-Det vggtdet.py](https://github.com/yangcaoai/VGGT-Det-CVPR2026/blob/73ea4cf94a05fa50279ee0b6095fb06f6d042dac/projects/VGGTDet/vggtdet/vggtdet.py): attention_guided_prob_fps starts with highest normalized attention; following points maximize normalized attention plus lambda times nearest selected-point distance. Original coordinates are3D points. We independently implement deterministic2D patch selection within original detection/context, exclude pad centers, cap context and add TRAIN-learned historical-query relevance. AG-A/B/C are distinct ablations; no random token sampling or3D detector is substituted.

[Pinned aggregator](https://github.com/yangcaoai/VGGT-Det-CVPR2026/blob/73ea4cf94a05fa50279ee0b6095fb06f6d042dac/vggt/models/aggregator.py) selects frame block0 attention and averages heads/query positions, retaining key saliency after patch_start_idx. This is incoming attention, not an identity probability. Our official-backbone hook calculates the same Q/K/RoPE statistic in chunks; max absolute error versus the actual pinned Det attention on real inputs is 9.976793080568314e-08. It does not write backbone output tensors.

## Actual QD mechanism and adaptation

[TransformerDecoder_Multilevel](https://github.com/yangcaoai/VGGT-Det-CVPR2026/blob/73ea4cf94a05fa50279ee0b6095fb06f6d042dac/projects/VGGTDet/detr3_models/transformer.py): separate four-layer channel projections in get_box_features; output task-query token -> task_head -> Softmax over layers -> weighted token memory -> decoder interactions. We use independent128-dimensional projections and a two-layer/four-head decoder. Global learned See-Query is mandatory control; per-history-track base+MLP conditioned See-Query is the hypothesis. AG/QD credit remains with VGGT-Det; no eight-layer/3D detection head/PointNet/FPS CUDA or MMDetection3D import.

## Layer/attention/preprocessing contracts

Official VGGT a288dd0f14786c93483e45524328726ab7b1b4ce returns24 positions with uncached None values; Det 73ea4cf94a05fa50279ee0b6095fb06f6d042dac returns a compact list for4/11/17/23. explicit_layer_map requires the correct explicit scheme. Real images[1,2,3,518,518], RGB[0,1], exactly one official internal normalization; patch_start_idx5 excludes camera/register tokens. Each retained layer[1,2,1369,2048]; attention[1,2,1369]. PadMapping round trips, original-coordinate row-major patch orientation and zero/tiny/padded regions are tested. No dense maps/full attention are persisted; only32 diagnostic positions/view remain outside Git.

## Numerical and weight compatibility

Both independent aggregators strictly load the original official aggregator checkpoint keys. Their32 retained positions/view are exactly equal at all four layers under matched math SDPA/deterministic cuDNN. The entire four-layer official output is exactly equal with and without the attention hook, as is a further plain repeat. Default FP16 backend probes did not repeat exactly even without the hook; those failures, unpacking error and raw diagnostics remain archived. The bounded backend check was documented before execution in TRACKVGGT_NUMERICAL_DIAGNOSTIC.md. New teacher extraction scopes the verified backend locally; it changes neither GMT settings nor weights. This is numerical interface evidence on one group, not a speed/identity result.

## Sources and licenses

VGGT-Det repository root LICENSE is MIT (copyright2024 yangcaoai); upstream-derived layers also carry Meta/Apache notices. Official VGGT LICENSE.txt is the separate VGGT License v1 dated July29,2025, with attribution, redistribution and Acceptable Use conditions. The official weights remain under their applicable official terms; the Det MIT root is not a replacement license. We reuse authorized local original weights, no indoor3D detection checkpoints. No commercial permission or deployment claim is inferred. External sources are read-only, SHA-pinned in the machine audit; no wholesale source copy.

## Remaining gates

Actual hard-event collection, all adapter edge cases, AG ranking/size/crop experiments, QD/query-shuffle, temporal cycles, conditional KD and complete deployment remain required. Phase0 success is not TEACHER_SUCCESS/STUDENT_SUCCESS/SCIENTIFIC_SUCCESS/PAPER_READY.

## Subsequent executed scope

The Phase0 interface/architecture paragraphs above describe prerequisites at
that time. Actual causal native data, all56 registered AG/QD/crop/temporal
fits and independent paired audits have since completed. The screening gate
is NO_UNIQUE_VGGT_TEACHER_EVIDENCE; IRGD/conditional trained students and
learned full-video deployment are unexecuted. Fresh native system work and
final audit remain separate; see the current teacher/system/final reports.
Implementation contracts are not a scientific-success or learned deployment claim.
