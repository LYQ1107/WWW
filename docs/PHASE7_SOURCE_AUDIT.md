# Native source and transition audit

All paths refer to the frozen Phase VI base, unless labelled Phase VII. The machine-readable
function inventory and file hashes accompany this document. No old outputs are rewritten.

| Component | Actual function and contract | Consequence / limit |
|---|---|---|
| Semantic decision | `jev_decision.legal_actions_for`, `_validate_actions`, `JEVDecisionController.forward` | Typed semantic actions, masked distribution; action tokens are not candidate IDs. State encoder is two LayerNorm/GELU layers; key/query scoring has no candidate-set attention by default. |
| Typed lifecycle | `jev_lifecycle.memory_features`, `reactivation_features`, `pack_typed_state`, `TypedJEVController.forward` | Relative 8/13 fields and adapters exist in research runtime. The 39,712-parameter controller was not jointly trained. Production REACT still uses legacy 64-state `_jev_reactivation_action`; relative hook absent. |
| Previous gates | `jev_lifecycle_gates.MatchThresholdGate.forward`, `decide_strict` | Deliberately rejects REASSOCIATE. It cannot be called a same-action attribution baseline without an explicit Phase VII adapter. |
| Baselines | `jev_baselines._prepare_inputs`, `_result`, `FixedThresholdPolicy.forward` and neural baselines | Same typed transport and masked probabilities, but fixed prefix accept feature is clipped. Phase VII explicitly uses raw proposal score 56 for accumulated GMT evidence. |
| Runtime | `jev_runtime.build_controller_from_checkpoint`, `JEVRuntimePolicy.decide`, `DecisionTraceWriter.write` | Loads controller-owned device, selects maximum probability with legal-order tie break; OFF preserves supplied GMT action. Trace context is diagnostic, not controller input. |
| Features | `jev_state.build_state_values`, `build_state_features`, `legacy_acceptance_threshold`, `count_track_history`, `association_window_length` | Feature 1 alternate is clipped, 56 raw score is unbounded. Track length multiplies legacy threshold unless NOT_MULT_THRESH. New baselines must disclose this information limitation. |
| Solver | `jev_assignment.constrained_hungarian` | One full matrix solve maximizes frozen scores; masked and nonfinite edges discarded. Does not lock ACCEPT rows. New IDs are handled later, not by this solver. |
| Production MATCH | `GTRRCNN._apply_jev_match_decisions` (250–487), called by `run_first_tracker_plus` and `run_global_tracker_plus` | Before commit: first A/R/NEW, NEW masks all row edges, R masks original edge, full re-solve, binary validation of all assigned A/R rows. Round-two feature uses newly assigned candidate/length, zero alternate and entropy, and R-or-original-unmatched flag. A row originally ACCEPT can move or be rejected. No R means no second validation. |
| Production lifecycle | `sliding_inference_GMT` → `get_asso` → `run_first_tracker_plus` / `run_global_tracker_plus` → `memory_bank` → `run_memory_tracker` → `_jev_memory_action` | MATCH existing IDs first, stale bank recovery for unmatched, fresh IDs for remaining, eligible bank writes. Birth/lost/refind and memory representation have separate dependencies. Whole-video input adapter does not imply future features are legal controller inputs. |
| Stale candidates | `memory_bank`, `jev_counterfactual_v2.promote_stale_bank_candidates`, `reactivation_candidates`, `build_reactivation_proposal` | Native possible-memory maturity, bounded observation bank, insertion order and history exclusion matter; no synthetic stale identities. |
| Mutable replay | `MutableGMTState.clone`, `sync_production_history_for_key`, `CachedPerceptionMutableAssociationV2.propose/resolve_actions/step` | Copies bank/history/IDs/Python trajectory RNG; proposal carries pre/post RNG and reused-score provenance. Legacy resolver's second validation is GMT threshold, not B2 learned validation. |
| Native MATCH bridge | `jev_phase6_native_match.NativeMatchResolver.resolve/native_call` | Calls actual production hook, replacing legacy resolver commits. Existing implementation performs a discarded legacy solve first; Phase VII avoids that duplicate while retaining native results. VALIDATION_ONLY keeps original pairs but still learned-validates A/R. |
| Replay lab | `jev_phase6_rollouts.NativeReplayLab.propose/choose/step/run`, `state_fingerprint`, `rebuild_memory_prefixes` | Live future policies and mutable history, exact prefix/RNG clone. Factual journals reconstruct prefix only; never use factual future actions after intervention. Module OUT globals point to Phase VI unless redirected *before* construction. |
| Label censor | `jev_phase6_label_contract.censor_unknown_target` | Unknown target has zero weight; avoids None==None recovery. Offline-only; do not call mutating censor on protected labels. |
| Previous fitting | `fit_jev_phase6_head.main` | Native manifest required, fixed20 epochs, 24 train/23 validation. It can rewrite censored source labels: never run against protected Phase VI sources. B2 is loaded into separate typed core; actual MATCH remains separate frozen B2. |

Exact production lifecycle order follows native implementation; no inferred three-module production
parity. Relative REACT integration remains **BLOCKED_NATIVE_RELATIVE_REACT_HOOK**. The Phase VII
MATCH bridge will require default B2 committed-ID and feature parity before heldout experiments;
an interface/source review alone is not a PASS for a complete mutated-state replay.

The previous `NATIVE_HELDOUT_MECHANISM_AUDIT.json` remains unchanged. Its pooled native B2 minus
VALIDATION_ONLY is +.31274676 HOTA / +.62122054 AssA, conditional on the B2 policy; this is
not evidence against ordinary rules with the same global solver. Video02 explains most of that
contrast, while video05's global re-solve slightly harms HOTA/AssA.

## ByteTrack: different algorithm

Read [ECCV 2022 paper, §3 / Algorithm 1](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf)
and [official update](https://github.com/FoundationVision/ByteTrack/blob/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99/yolox/tracker/byte_tracker.py).
High detections associate to tracked+lost Kalman-predicted tracks; low detections then associate
to remaining **Tracked** objects with IoU. Unconfirmed matching and high-score births follow;
lost tracks expire separately. Official LAP cost-limited matching is not this project's scipy
banned-edge Hungarian over the entire GMT proposal. Paper default high threshold is .6; our
predeclared .5 is an explicit fixed configuration, not an assertion about its paper default.
The cache's low-count audit concerns real cached detections; absence does not prove ByteTrack
would fail with the detector's discarded boxes available.

WHAT DID WE LEARN? First-round action equality is not commit equality. Global re-solve, assigned
candidate feature reconstruction, second validation and subsequent memory eligibility must all
be checked. The current code does not support a native relative three-question production claim.
