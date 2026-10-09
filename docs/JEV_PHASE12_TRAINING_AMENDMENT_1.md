# Pre-training comparison clarification

Frozen before any real MATCH fit or validation performance. Main v2 network and native semantics are unchanged.

Add `numerical_only`: exactly the FullVisualJev parameterization and all numerical/context/metadata inputs, with current/shared/historical visual tensors zeroed. This directly tests added visual information under identical reader capacity and supervision. The original state64/e12 models remain the historical architecture comparison, with their smaller capacities explicitly reported.

Original Phase X MLP/DeepSets/CandidateJEV preserve the exact original two-channel numerical network; a common 2->3 auxiliary consequence projection supplies multihorizon supervision. This adds nine parameters and is disclosed as an output adapter, not a redesign or a capacity-matched visual baseline. All models see the same executed H8/16/32 targets and masks. Same-visual ordinary controls retain the same full visual inputs and typed heads as FullVisualJev.

The first/long history token is the first resident native Gallery element. In GMT it can be overwritten by a bank promotion prototype through aliased Instances; it is not promised to remain the original birth image embedding. We preserve the actual native state, rather than reconstructing a supposedly pristine history.

Calibration fits a scalar temperature to the final joint preference on labeled validation support, with separate UNKNOWN certificates. It is not a learned abstain/DEFER/NEW classifier. The native common raw GMT threshold-offset defer remains frozen, and abstain confidence/margin rules remain those in v2. Same risk policy for controls. All temperatures and checkpoint choices freeze before online validation.

Tiny uses 1000 updates for CE/H32/joint across the three frozen seeds. Full training100 epochs all registered conditions. Unknown fitting failures cannot become fabricated negatives and do not restore the old all-model95% stop rule. No online validation or heldout result has been observed at this amendment.
