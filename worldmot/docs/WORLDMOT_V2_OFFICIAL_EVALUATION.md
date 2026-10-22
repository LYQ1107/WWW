# Official evaluation gate

Formal HOTA/DetA/AssA/IDF1/MOTA/IDSW/Frag and cross-view metrics must be
computed with the pinned TrackEval implementation on complete legal video
sequences.  `worldmot.evaluation.official_tracking_eval` locates TrackEval,
records its commit/path, and refuses to label the simplified v1 evaluator as
official.  Short clips and diagnostic metrics remain useful for debugging but
cannot be reported as official aggregate scores.

The current branch has no official PWQT score.  The result files use explicit
`NOT_RUN`/`BLOCKED` statuses instead of fabricated numbers.

