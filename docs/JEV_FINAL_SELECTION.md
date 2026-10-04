# Final selection and official-test gate

Architecture, hidden size, threshold baseline, MLP baseline, calibration
temperature, and all hyperparameters are selected from policy train/validation
sequences only. The selection record must include:

- GMT checkpoint SHA256;
- code commit and state schema version;
- counterfactual engine and utility definition;
- horizon;
- selected JEV architecture and hidden size;
- selected calibration method;
- selected threshold and MLP baselines;
- all training hyperparameters;
- policy split SHA256 and sequence hash.

`reproduction_tools/create_final_selection_lock.py` writes this atomically to
`manifests/FINAL_SELECTION_LOCK.json` and refuses an unverified policy split.
No official-test reader may run before this file exists. The official suite
then runs every selected method once under the same checkpoint/data/evaluation
conditions.

GO requires positive, stable persistent-state evidence after the stronger
threshold and parameter-matched generic MLP controls, acceptable runtime
overhead, and no online future-GT access. Failure of a key control produces a
NO-GO result rather than a cherry-picked claim.

