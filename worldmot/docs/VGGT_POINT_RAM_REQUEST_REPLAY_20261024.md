# VGGT point-request RAM replay audit (2026-10-24)

This is a TRAIN calibration engineering audit. It does not change the registered point-association grid, the detector cache, the VGGT weights, the point sampler, or any TEST selection rule.

## Why this was added

The first calibration partition showed that request-cache I/O and image/query hashing dominated wall time after the genuine VGGT forward. The old disk cache remains the canonical persisted format. `scripts/ram_point_request_cache.py` adds a bounded, read-only RAM layer over that exact namespace, with asynchronous atomic writes for new requests. The request signature still contains the unchanged core source hash, the original disk-cache helper hash, the official checkpoint hash, precision, RGB bytes, camera/timestamp order, and exact FP32 query bytes.

## Real persisted-request audit

One published request from the existing VGGT-1B TRAIN calibration cache was loaded into the new layer with a backend that raises on any forward. The four arrays were bitwise equal to the published NPZ and were marked read-only. The replay reported `forward_count=0`.

This is a persistence/replay audit only. It is not an end-to-end FPS measurement and does not use TEST data.

## Queue migration

The interrupted disk-bound calibration partition was preserved under `calibration_v1`. The idle full-video waiter and the two calibration workers were stopped with `SIGINT`, leaving their status and partial artifacts intact. The new workers run under `calibration_v2`, reuse only the old verified request root, and use the same six-scene TRAIN validation split and 648-window budget. GPU 3 and GPU 7 are the two point-scan workers; the GMT detector/object-cache, GMT training, 2D controls, and original baseline jobs remain independent.

Every window is published `COMPLETE` only after pending asynchronous writes are flushed. The queue still refuses to select or evaluate TEST until both complete partitions, full six-scene validation, and the existing integrity gates pass.

The machine-readable audit is in `results/vggt_point_ram_replay_audit_v1.json`.
