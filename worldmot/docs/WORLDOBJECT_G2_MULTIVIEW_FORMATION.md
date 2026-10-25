# World Object G2 — TEST multi-view formation

The joint builder enforces one observation per camera per world-object
hypothesis and abstains when geometry is not reliable. It does not merge by
nearest centroid alone.

Across TEST:

- hypotheses: 399,470;
- two-view fused hypotheses: 149,113;
- single-view hypotheses: 250,357;
- correct fused hypotheses: 118,274;
- wrong fused hypotheses: 30,690;
- correct/(correct+wrong): 79.3977%; wrong/(correct+wrong): 20.6023%;
- candidate pairs evaluated: 5,559,143.

The diagnostic threshold curve is not used to select TEST parameters. At the
registered diagnostic threshold 0.5 it gives same-person recall 75.2339% and
different-person false-merge rate 3.4485%; this is a formation diagnostic,
not a substitute for official MOT scoring.

