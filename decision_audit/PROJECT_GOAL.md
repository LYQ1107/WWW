# GMT decision-formulation feasibility audit

This branch is an independent feasibility prototype created from
`d160c369e4739a6d68a484cb7a1203f6f5aea773`. The previous causal artifacts
under `causal/`, `causal_followup/`, and `causal_v2/` are retained unchanged.

The question is whether GMT's strictly online global-association choice can be
represented as a calibrated dynamic-candidate decision:

```text
current observation + active Global-ID candidates
    -> calibrated P(ID_1 ... ID_K, NEW)
    -> one-to-one assignment
```

The audit compares the released GMT choice, temperature-calibrated GMT, a
candidate-independent linear scorer, a candidate-independent MLP, and a
permutation-equivariant SetChoice head. Verification is an uncertainty
diagnostic only. This branch does not implement COMMIT, SOFT_COMMIT,
QUARANTINE, RECOVER, delayed decisions, history rewriting, or RLCD.

All records are built from the VisionTrack **train** scenes. Scene splits are
frozen before results: 70% JEV-train, 10% calibration, and 20% dev using
SHA256 scene ordering. GT is used only after online record construction to
build supervision labels and offline evaluation. VisionTrack test GT is not
read, and no final-test evaluation is run.

The released checkpoint has already seen the complete VisionTrack training
set. Results are therefore named **DECISION-FORMULATION FEASIBILITY
PROTOTYPE**, not an independent paper result.

The offline gates are:

1. frozen candidate recall reaches 95% at the smallest tested K;
2. Oracle Choice has practical tracking headroom;
3. SetChoice beats GMT on hard-recoverable decisions and is compared against
   Linear and MLP under the same records, calibration, and seeds;
4. at least two of three seeds show SetChoice > MLP by at least one percentage
   point and one of NLL/Brier improves.

The final classification is one of `JEV_DECISION_GO`,
`LEARNED_ASSOCIATION_ONLY`, or `NO_DECISION_HEADROOM`. It is not selected in
advance.
