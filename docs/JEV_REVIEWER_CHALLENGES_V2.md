# JEV reviewer challenges — v2 evidence plan

This document defines the evidence required before making a causal claim about
JEV. The v1 `frozen_evidence_mutable_gmt_state_v1` labels remain proxy
evidence only.

| Reviewer challenge | Required evidence | Current status |
| --- | --- | --- |
| Is the gain only a nonlinear threshold? | State-conditioned, question-conditioned, and nonlinear full-state threshold controls with the same data, labels, supervision, and three seeds | Code present; formal H×seed suite pending |
| Is the gain only a generic MLP? | Fixed-slot, independent-head, shared-head, question-conditioned, parameter-matched, and action-conditioned generic MLP controls | Code present; formal H×seed suite pending |
| Does the controller choose identities? | GMT proposes identities; JEV chooses typed actions only | Runtime contract implemented |
| Is REASSOCIATE just top-2? | Mask rejected edges and run one complete constrained Hungarian solve | Implemented and regression-tested |
| Is there future/GT leakage? | Online trace has no future GT; labels and official evaluation are separated | Contract tests pass; formal trace audit pending |
| Are counterfactuals causal? | Same frozen perception, branch-local association/state/memory dynamics | v2 protocol, cached adapter, and shard execution in progress |
| Was official test used for selection? | Sequence split manifest and canonical `FINAL_SELECTION_LOCK.json` before official test | Fail-closed gate implemented; lock not yet issued |
| Are calibration and confidence comparable? | Val-only temperature fitting for every candidate plus NLL/Brier/ECE/risk-coverage | Protocol runner added; formal suite pending |
| Is the result persistent-state reasoning? | Horizon sweep, memory/reactivation ablations, and state-transition metrics | Analysis protocol documented; runs pending |

## Evidence labels

- **Contract test**: CPU invariant or schema test; it does not establish a
  tracking result.
- **Proxy**: useful for scheduling and debugging, but not a final causal
  result.
- **Formal**: fixed checkpoint, frozen code/data manifest, policy-val-only
  selection, and one-time official-test evaluation.

No formal GO claim is allowed while the v2 cache uses the cosine contract
backend, while same-GPU OFF equality is unverified, or while the final-selection
lock is absent.
