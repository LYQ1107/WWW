WorldMOT V5: resource observations for timing interpretation

At 2026-10-27 10:40:06 UTC, a read-only NVIDIA query observed one additional compute process using 17,427 MiB on B3's GPU. B2 and B3 retained their original training PIDs. A preceding three-second CPU sample showed continued computation and recent finite-loss logs. The jobs continue with their frozen parameters and budget. This observation does not establish how much of the scene latency was caused by GPU sharing.

TRAIN elapsed time is not a method FPS result. New-model cached replay and real RGB timing remain pending. The resource observer adds evidence about the environment without changing the training, tracking, scoring, input scope or the requirement for all three final weight gates before new-model TEST.

The observer queries NVIDIA device/application metadata every five seconds. Goal processes are identified by PID plus process start tick, including descendants of the verified controller. PID reuse cannot turn an unrelated process into a goal process. Other workload is represented by counts and memory totals; unrelated PIDs, user names, command lines and project paths are omitted. The observer requests no GPU allocation, reads no GT and controls no model process.

Its source hash and protocol are published before the continuous observer starts. It runs for at most eight hours or until the research delivery receipt appears. The private runtime log retains timestamps, sample validity, device names/UUIDs, goal process identities and observed contention. Invalid or missing query results cannot establish an idle device. Five-second snapshots can miss activity between samples, so they do not prove continuous GPU exclusivity.

Timing windows with observed other workload cannot support an uncontended speed ranking. Actual latency remains reportable with its environment label. Tracking outputs and quality metrics remain in the registered study regardless of occupancy; the observer introduces no model selection, parameter tuning, extra seed or optimization trial.

Three generated-only checks passed: process ancestry and PID reuse, distinction between shared and idle devices, and invalid-query handling. A separate real query smoke check confirmed attribution of the two current training processes and detected B3's observed sharing. These checks are separate from the earlier 304 core and 85 analysis checks; those suites were not rerun for this instrumentation change.

[Observer protocol](../results/v5_gpu_occupancy_protocol.json), [generated checks](../results/v5_gpu_occupancy_checks.json), and [real query smoke evidence](../results/v5_gpu_occupancy_smoke.json) preserve the source hashes and observation scope. The frozen model source fingerprint remains `c4f9d1c0b2a290062aba52ae7c0d322176ae5376f9c85688f74d89159e16f733`.

First archived continuous prefix: 2026-10-27T10:46:48.446993+00:00 through 2026-10-27T10:59:44.054780+00:00. All 156 queries were valid. The B3 device showed unrelated GPU context occupancy in 156/156 samples, with up to 17,427 MiB from that workload. The B2 device showed such occupancy in 151/156 samples, with up to 2,037 MiB. These are process-context observations, not proof of simultaneous active kernels or a causal latency estimate. Full lists of goal processes are retained as well; overlapping goal workers on the same device also prevent an uncontended single-worker speed claim.

The preserved prefix and its source/launch/protocol records were archived on the actual NAS and fully read back. [Prefix coverage and archive proof](../results/v5_gpu_observation_progress.json) preserves exact hashes. The observer continues after this snapshot.

The newly completed B3 TRAIN epoch 8 used 3923.9 seconds of summed scene compute wall time. Its recorded occupancy overlaps only part of that epoch. Training elapsed time remains separate from cached inference and real RGB-to-ID speed.

Second archived continuous prefix: 2026-10-27T10:46:48.446993+00:00 through 2026-10-27T12:23:22.926706+00:00. Of 1159 samples, 1159 were valid and 0 invalid. B3's GPU had another process context in 1159/1159 samples; B2's GPU had another context in 291/1159 samples. These are sampled contexts, not measured GPU kernel utilization or proof of latency causation. Samples can miss activity between queries.

The observer does not modify model inputs, hyperparameters or optimization. The B2 and B3 optimizers continue updating their learned weights during their normal training. [Second prefix and NAS readback proof](../results/v5_gpu_observation_progress_02.json) retains exact timestamps and hashes.
