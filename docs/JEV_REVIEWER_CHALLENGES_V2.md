# JEV reviewer challenges — v2 evidence plan

This document defines the evidence required before making a causal claim about
JEV. The v1 `frozen_evidence_mutable_gmt_state_v1` labels remain proxy
evidence only.

| Reviewer challenge | Required evidence | Current status |
| --- | --- | --- |
| Is the gain only a nonlinear threshold? | State-conditioned threshold and question-conditioned threshold with the same data, labels, and budget | Implemented; formal v2 training pending |
| Is the gain only a generic MLP? | Question-conditioned generic MLP, parameter-matched shared heads, and action-conditioned scorer without question | Implemented; capacity suite pending |
| Does the controller choose identities? | GMT proposes identities; JEV chooses typed actions only | Runtime contract implemented |
| Is REASSOCIATE just top-2? | Mask rejected edges and run one complete constrained Hungarian solve | Implemented and regression-tested |
| Is there future/GT leakage? | Online trace has no future GT; labels and official evaluation are separated | Contract tests pass; formal trace audit pending |
| Are counterfactuals causal? | Same frozen perception, branch-local association/state/memory dynamics | v2 protocol and builder implemented; GMT association adapter pending |
| Was official test used for selection? | Sequence split manifest and `FINAL_SELECTION_LOCK.json` before official test | Gate scripts implemented; lock not yet issued |
| Are calibration and confidence comparable? | Val-only temperature fitting plus NLL/Brier/ECE/risk-coverage | Implemented; formal suite pending |
| Is the result persistent-state reasoning? | Horizon sweep, memory/reactivation ablations, and state-transition metrics | Analysis protocol documented; runs pending |

## Evidence labels

- **Contract test**: CPU invariant or schema test; it does not establish a
  tracking result.
- **Proxy**: useful for scheduling and debugging, but not a final causal
  result.
- **Formal**: fixed checkpoint, frozen code/data manifest, policy-val-only
  selection, and one-time official-test evaluation.

No formal GO claim is allowed while the v2 cache uses the cosine contract
backend or while the final-selection lock is absent.

