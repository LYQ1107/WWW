# World Object G0 — TEST world-coordinate alignment

Scope: independent WorldMOT World Object experiment, TEST split, 22 scenes and
29,019 synchronized timestamps. The immutable GMT candidate cache and verified
VGGT-Ω 512 geometry are used; GT is diagnostic-only and is not consumed by the
runtime.

The dual-camera static-background alignment (G0-B) succeeds on 99.9207% of
timestamps (timestamp-weighted), with a 0.0793% object-build blocked rate. The
median per-scene residual-p95 is 0.00940 in the cached world-coordinate unit;
the timestamp-weighted mean of per-scene residual-p95 medians is 0.01065. The
median round-trip transform error is `4.98e-16`, and the median scale over
scenes is stable. The original first-camera alignment (G0-A) succeeds on
97.5671% of timestamps.

G0 is therefore not the dominant failure for the complete TEST cache. The
remaining failure cases are retained in `G0_world_alignment_all22.csv` and are
not silently dropped from later routes.

