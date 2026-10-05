# Final selection and official-test gate

Architecture, hidden size, threshold baseline, MLP baseline, calibration
temperature, horizon, and all hyperparameters must be selected automatically
from policy train/validation sequences only. The required protocol is three
seeds over H=1/8/16/32 and the complete threshold/nonlinear-threshold and
generic-MLP control family under equal supervision. The selection record must
include:

- GMT checkpoint SHA256;
- code commit and state schema version;
- counterfactual engine and utility definition;
- horizon;
- selected JEV architecture and hidden size;
- selected calibration method;
- selected threshold and MLP baselines;
- all training hyperparameters;
- policy split SHA256 and sequence hash.

It must also include every candidate's validation NLL/accuracy, seed aggregate,
calibrated checkpoint, feature-audit result, and deterministic selection rule.
Official TEST is not an input to any of those files.

`reproduction_tools/create_final_selection_lock.py` writes this atomically to
`manifests/FINAL_SELECTION_LOCK.json` and refuses an unverified policy split,
noncanonical checkpoint, or missing PASS selection-protocol digest.
Official TEST counterfactual manifests are additionally bound to the final
lock's SHA256 digest; pre-lock diagnostic manifests are rejected on restart.
No official-test reader may run before this file exists. The official suite
then runs every selected method once under the same checkpoint/data/evaluation
conditions.

GO requires positive, stable persistent-state evidence after the stronger
threshold and parameter-matched generic MLP controls, acceptable runtime
overhead, and no online future-GT access. Failure of a key control produces a
NO-GO result rather than a cherry-picked claim.

The historical `model_4500` lock is proxy screening evidence and can never
authorize official TEST access. No formal GO claim is allowed while the strict
same-GPU OFF gate or this canonical final-selection lock is absent.
