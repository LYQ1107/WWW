# FPS measurement contract

FPS is reported as synchronized camera pairs per second.  Timers include
queueing and synchronization and are split into RGB read/decode, detector,
Ω forward, pointmap/object sampling, world alignment, query decoder, memory,
projection, and final output.  Each component records P50 and P95; model-only
latency is never substituted for end-to-end latency.  Four-card throughput is
reported separately from single-stream latency.

The first registered baseline is one full Ω forward per synchronized pair.
Intervals 5 and 10 may be tested only after the current-frame detector,
camera update, and legal position prediction are implemented; a stale person
depth map cannot be reused as a current observation.  The v2 query-only
benchmark in `results/worldmot_v2_fps.csv` is explicitly not a system FPS
claim.

