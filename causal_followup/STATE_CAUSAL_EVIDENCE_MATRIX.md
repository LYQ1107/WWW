# State-causal evidence matrix

This matrix is completed from the immutable first-round audit and the
event-isolated C1b/C2b replay. The numerical C1b/C2b rows are filled by
`causal_followup/analyze_followup.py` after both result CSVs finish.

| Experiment | Intervention | Isolation / estimand | Result | Gate interpretation |
|---|---|---|---|---|
| C1 original assignment correction | Current wrong ID changed to the frozen correct candidate | First-round long rollout; confounded by later state changes | First-round effect was positive at all horizons; no expected negative recovery | Assignment-only correction was not supported |
| C1b-B quarantine | Current output set to correct ID; current observation excluded from future persistent history | 71 events; 66 estimable at +5/+10/+20; 19 scene-end rows and 80 current-target-occupied rows skipped | +5 Δ −0.081 pp, 95% CI [−0.168, 0.000] pp; +10 Δ −0.082 pp, CI [−0.239, 0.000] pp; +20 Δ −0.081 pp, CI [−0.189, 0.000] pp | Direction is weakly favorable but intervals include zero; no state-repair gate |
| C1b-C oracle state repair | Repair wrong-ID history over the preceding 20 frames, then commit current observation to correct ID | 71 events; 60 estimable at +5/+10/+20; 180 repair-conflict rows and 20 target-occupied rows skipped | +5 Δ −0.091 pp, 95% CI [−0.516, +0.494] pp; +10 Δ +0.182 pp, CI [−0.133, +0.787] pp; +20 Δ −0.090 pp, CI [−0.208, +0.017] pp | No significant repair benefit; no state-repair gate |
| C2 original compound injection | Multiple injected wrong commits in one rollout | First-round long rollout; later injections could be affected by earlier ones | +5 Δ approximately +0.376; 95% CI [0.334, 0.419] | Strong positive direction, but not isolated |
| C2b isolated single injection | One correct-to-wrong commit, then ordinary GMT continuation | 256 events; 242 estimable at +5/+10, 241 at +20; 280 target-occupied rows and 8 scene-end rows skipped | +5 Δ +0.074 pp, 95% CI [−0.011, +0.139] pp; +10 Δ +0.001 pp, CI [−0.048, +0.052] pp; +20 Δ −0.047 pp, CI [−0.121, +0.027] pp | Positive direction at +5 but interval crosses zero; isolated injection gate fails |

## Decision rule

`STRONG_GO` requires isolated C2b injection to increase future error at at
least one target horizon with a paired 95% interval excluding zero and at
least two of the three scenes positive at that horizon. It also requires C1b-B
or C1b-C to reduce future error at +5, +10, and +20 with paired intervals
excluding zero and at least two scenes negative at every target horizon.

The observed result is `STOP_NO_STRONG_CAUSAL_GATE`: neither the isolated
C2b condition nor the C1b repair condition passes. No JEV-GMT prototype is
authorized from this audit.
