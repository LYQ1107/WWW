# FPS benchmark protocol

The timer separates `geometry`, `model_only`, `world_alignment`, `instance_build`, `identity_memory`, `reprojection`, and `total`. It reports mean, P50, P95, host RSS, and CUDA allocated/reserved/peak memory. `pipeline_fps` is the reciprocal of the measured synchronized-pair total; it is never renamed as camera-image throughput or four-GPU aggregate throughput.

The first benchmark should use one process on one explicitly free GPU, warm up once, then run continuous synchronized frames. Any parallel multi-GPU experiment must report per-device latency separately.
