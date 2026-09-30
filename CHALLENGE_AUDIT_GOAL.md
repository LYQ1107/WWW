# GMT Challenge Audit Goal

This worktree is a diagnosis-only audit of the completed released-code
VisionTrack reproduction. It is isolated on branch `challenge-audit` and must
not alter the release-reproduction worktree or `main`.

## Fixed baseline

- Source commit: `dfa9ca8e0b8da5c2af89ef3d3ac4f9d991162ebb`
- Stage2 checkpoint: `audit/links/stage2_model_20000.pth`
- Baseline metrics: CVMA 75.630, CVIDF1 78.740, MOTA 80.350, HOTA 66.645,
  IDF1 81.201, AssA 67.706
- Baseline prediction manifest: `audit/manifests/baseline_prediction_manifest.json`
- Full provenance: `audit/manifests/baseline_provenance.json`

## Scope

The audit covers only A0 seed/jitter sensitivity, A1 quality stratification,
A2 history-policy sensitivity, A3 history-length sensitivity, A4 error
persistence and cross-view cascade, and A5 feature/retrieval diagnostics.
The audit may read the fixed checkpoint, test annotations, images, and baseline
predictions. It may write diagnostic tables, plots, and manifests under
`audit/`.

No formal training, fine-tuning, checkpoint modification, ground-truth
modification, threshold tuning, association-formula change, new model module,
new dataset, DIVO/WILDTRACK run, or automatic follow-on experiment is allowed.
The official released metrics remain the baseline reference; diagnostic
matching and audit metrics are clearly labeled as such.

## Required stop point

After A0-A5, provenance, evidence matrix, and GO/NO-GO report are written, the
worktree stops and waits for an explicit next instruction.
