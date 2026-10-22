# Registered ablations

The planned matched-capacity comparisons are:

1. 2D appearance-only learned association.
2. world xyz-only association.
3. xyz plus MLP.
4. xyz plus Transformer without persistent state.
5. v1 two-stage cross-view fusion followed by temporal memory.
6. joint multi-view Persistent World Query (PWQT).

All variants must consume the same detector outputs, Ω input pair, RGB
timestamps, scene reset rules, and legal TrackEval protocol.  Any variant that
uses random or unadapted weights is a smoke test and cannot enter the formal
table.

