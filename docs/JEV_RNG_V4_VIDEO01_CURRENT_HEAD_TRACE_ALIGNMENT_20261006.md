# Video 01 current-head trace alignment

This diagnostic is not a tracking result. It records why the old video1
counterfactual records cannot be used as the final corrected-v4 training or
runtime artifact.

The current-head formal OFF replay was generated from the frozen perception
cache and `model_20000.pth` with the explicit branch-local trajectory RNG.
Its trace contains **8,995** events:

| question type | legacy trace | current-head trace |
| --- | ---: | ---: |
| `MATCH_DECISION` | 4,524 | 4,524 |
| `MEMORY_DECISION` | 4,305 | 4,297 |
| `REACTIVATION_DECISION` | 167 | 174 |
| total | 8,996 | 8,995 |

When the legacy trace is supplied only as a reference action layout, the
current replay compares 8,884 action records and reports **24 OFF-action
mismatches**. This is the same class of semantic state/action divergence
already diagnosed on video06/video07; it is not a numerical tolerance issue.

Therefore the old video1 builder output remains preserved but blocked for final
use. A separate current-head records rebuild has been started on an idle GPU
while the previously running GPU8 video1 builder continues untouched. The new
records must pass provenance, exact three-question key parity, runtime feature
parity, and reactivation candidate parity before they can enter compact dataset
construction or policy training.

The raw trace and its manifest are runtime artifacts outside Git. Their paths,
SHA256 values, and all gate decisions are recorded in
`reports/JEV_RNG_V4/VIDEO01_CURRENT_HEAD_OFF_TRACE_ALIGNMENT.json`.

