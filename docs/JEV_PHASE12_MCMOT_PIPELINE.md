# Phase XII native online pipeline

Default `VISUAL_JEV_ENABLED=false`. OFF executes the original code with no new
tensor build, RNG draw or altered solver. SHADOW constructs typed observations
and predicts, then returns original native actions. MATCH_ONLY changes only
active identity matching; actual stale bank and memory remain frozen native
rules. FULL_LIFECYCLE fails closed unless independently qualified checkpoints
for both tasks exist; random heads never mutate online state.

1. Native current perception and fresh GMT scores are available. Build shared
   state from current ReID and prefix Gallery; generate one MATCH question per
   detection and full lawful active options plus DEFER.
2. Score active identities, solve per-camera capacity globally. Risk fallback
   applies to the complete graph component formed by shared legal identity
   edges. All fallback rows in a component jointly use native rectangular
   Hungarian then threshold. DEFER leaves -1 for actual unmatched handling.
3. The original native `memory_bank` forms real stale candidates using actual
   bank eligibility and GMT scores. Only these unmatched rows get REACTIVATE
   questions. Frozen native proposal stays in force until qualified data exists.
4. Final identity is committed and hit count updated. Rebuild MEMORY question
   from the committed ID/current vector/true latest Gallery; WRITE concatenates
   the actual Instances exactly once, KEEP leaves Gallery intact. Mandatory
   birth initialization and bank bookkeeping preserve original semantics.
5. Next frame recomputes GMT candidates/scores on the mutated tracker history.

Visual projection caches may reuse immutable feature vectors. The shared memory
is tied to the exact state fingerprint and cannot cross a commit. No future
visual/action maps, labels or evaluators are imported by this runtime package.
Repeated questions read one immutable shared memory; contexts are not cached
across stages. Frozen feature reuse avoids rerunning the perception backbone.

Production integration is in `gtr/modeling/meta_arch/gtr_rcnn.py` through
`visual_jev_mcmot/lifecycle_controller.py` and `native_action_adapter.py`.
`jev_native_state.py` remains the actual prefix/fork adapter. Offline evaluators
are invoked after actor outputs have been saved, in separate reproduction tools.
