# G3 full TRAIN persistent-world replay summary

The causal persistent bank was replayed over all 24 TRAIN scenes using no GT at inference. GT was used afterward only to count identity diagnostics:

- Identity observations: **362,703**.
- Predicted global-ID switches: **119,721** (33.01% of identity observations).
- Wrong world-object merges: **34,831**.
- Oracle identity switch count: 0 by construction of the diagnostic oracle.

The error is strongly scene dependent: Scenes 3, 12, 13, 17–21, 23, and 24 are difficult, while Scenes 1, 5, 8, and 9 are much more stable. This is the untrained W3 persistent-bank diagnostic, not a final W4 claim; the learned W4 and W5 routes must be evaluated on the common TEST cache before any MOT conclusion.
