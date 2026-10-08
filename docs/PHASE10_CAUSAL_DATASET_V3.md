# Native candidate dataset v3

Status: FROZEN_ELIGIBLE. 226 frozen selected rows; all 162 old causal events and 840 original branches re-executed from actual native prefixes. Verified corrections: {'train': 74, 'validation': 59, 'supplemental': 46, 'supplemental_validation': 46}. Changed branch horizon labels: 53. Original Phase V–IX evidence remains unchanged.

Correctness uses only pre-intervention confirmed anchors. Unknown identities are masked. Utilities remain unavailable for unexecuted or censored candidates; no zeros replace missing Q. Single-candidate ranking excludes REASSOCIATE/joint policy effects and conflicting utilities for the same identity under different complete assignments. All such effects remain in raw branch evidence. Main NEW is fixed because no genuine positive NEW supervision exists. Original group names and equal total group weights are preserved; overlapping windows are separately reported as dependent bundles.

The state and data gates are independent. Passing serialization does not establish model performance. The compact manifests bind all large tensors and branch records on the server by SHA256.
