# GMT challenge audit: code-level review

This document describes the released source at commit
`dfa9ca8e0b8da5c2af89ef3d3ac4f9d991162ebb`.  The audit worktree is on the
`challenge-audit` branch.  All changes made for this audit are guarded by
`GMT_AUDIT_*` variables; with those variables unset, the original execution
path is retained.

## Global association: `run_global_tracker_plus`

In `gtr/modeling/meta_arch/gtr_rcnn.py`,
`run_global_tracker_plus` receives the association head output for the new
view/frame and applies `_activate_asso`.  The resulting `asso_nonk` has one
row per current detection and one column per historical detection in the
active temporal/view window.  Its entries are the association evidence from
the current detection to each historical detection after the GMT association
activation function.

The historical detections carry `ids`, the current global track IDs.  The
code constructs:

```python
unique_ids = torch.unique(ids)
id_inds = (unique_ids[None, :] == ids[:, None]).float()
traj_score = torch.mm(asso_nonk, id_inds)
```

`id_inds` is an `Np x M` one-hot membership matrix: row `p` describes which
of the `M` global IDs owns historical observation `p`.  The matrix product
sums every historical association score belonging to each candidate global
ID.  Therefore `traj_score[i, j]` is aggregated historical evidence for
current detection `i` and global ID `j`; it is not a learned reliability
weighted sum.

When `NOT_MULT_THRESH` is false, the acceptance threshold is
`OVERLAP_THRESH * id_inds[:, j].sum()`.  The threshold grows linearly with
the number of historical observations assigned to the ID.  The evidence and
the threshold thus both scale with history count, while each observation's
coefficient is one.  There is no detector-confidence, area, blur, occlusion,
view, age, or feature-consistency weight in this path.

The same uniform membership aggregation is present in the first-frame and
memory fallback trackers.  The challenge audit adds an optional history
weight vector only when `GMT_AUDIT_HISTORY_POLICY` is set; it does not change
the default code path.

## GMT memory bank: `memory_bank`

`memory_bank` stores the historical `Instances` for an ID in
`id_reid_dict`.  Once an ID becomes eligible for the bank, the released code
executes:

```python
for i in range(thred):
    sum_reid += id_reid_dict[id][-1-i].reid_features
aver_reid = sum_reid / thred
```

With the released VisionTrack test configuration, `WITH_BANK=True` and
`BANK_SIZE=10`, so the most recent ten ReID observations are averaged with
equal coefficients.  The calculation does not inspect detector score,
bounding-box area, blur, occlusion, view identity, feature consistency, or
observation age.  The bank is therefore **quality-agnostic / uniformly
averaged**.  This audit turns the bank off for A2 so that the causal history
intervention isolates the sliding-window evidence rather than mixing it with
the separate memory-bank fallback.

## Association feature extraction: `_forward_asso`

In `gtr/modeling/roi_heads/gtr_roi_heads.py`, inference first filters proposal
instances by `objectness_logits`, then obtains `proposal_boxes`.  The released
code immediately calls:

```python
proposal_boxes = self.jitter_bboxes_center(proposal_boxes)
pool_features = self.asso_pooler(features, proposal_boxes)
reid_features = self.asso_head(pool_features)
```

`jitter_bboxes_center` uses `torch.rand_like` for a random shift magnitude and
`torch.randint` for direction and sign.  `_forward_asso` is on the inference
path used by `CustomRCNN.inference`, so released inference is stochastic even
when the model is in evaluation mode.  The audit does not remove this by
default.  `GMT_AUDIT_DISABLE_TEST_JITTER=1` is an explicit diagnostic switch
that skips only this jitter while leaving the official default unchanged.

The optional `GMT_AUDIT_DUMP_FEATURES=1` switch records the raw appearance
output before the spatial/temporal concatenation and the normal fused
`reid_features` after it.  Features are written as float16 tensors in binary
files, never into the COCO JSON.  The dump is placed after tracking and short
track filtering, immediately before postprocessing, so it observes the same
predicted global IDs used by the baseline output.

## Audit interpretation limits

The observation-to-GT matching in A1/A4/A5 is a diagnostic Hungarian match
with IoU >= 0.5.  It is not the official HOTA, IDF1, CVMA, or CVIDF1 matching
procedure.  Test GT is used only descriptively for the predeclared audit
tables and is never used to select a model, threshold, or proposed method.
