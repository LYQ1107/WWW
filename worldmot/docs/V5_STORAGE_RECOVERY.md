# V5 storage failure and checkpoint recovery

Snapshot UTC: 2026-10-26T21:29:47.984868+00:00. This is an engineering recovery record; V5 official TEST metrics are still pending.

At about 20:48 UTC the NAS transport failed with errno 103/107. All three original training workers and three in-flight reference scenes exited with actual tracebacks. No healthy model worker was terminated to recover them. Completed B0 videos and failed logs/partial outputs were retained.

The three scene-boundary checkpoints were copied with matching SHA-256 into a local durable working directory. The unchanged frozen trainer used `--resume` and restored weights, optimizer and RNG. Subsequent complete-scene progress confirms that training continued. B3 checkpoint payload contains one more completed scene than the stale progress sidecar written immediately before the disconnect; the verified checkpoint payload is authoritative.

| Variant | Restored checkpoint | SHA-256 | Current completed selection epochs |
|---|---|---|---|
| B2 | resume_00059.pt | `4e8ad42f408c38e100bdaf896f775756c3e6723bd3ae1d503dc940befa161faf` | 2 |
| B3 | resume_00056.pt | `cb723af6997c46d4f2e7ca860686789fafa45615dd64cfd6b6c704734f657d28` | 2 |
| B4 | resume_00071.pt | `c0f4cefac295d6decf4653a0924b40854628031651d19fba75c7394bb390cb94` | 3 |

Frozen training/inference source: `c4f9d1c0b2a290062aba52ae7c0d322176ae5376f9c85688f74d89159e16f733`. Training scope, authorization and calibration bytes are unchanged. The 12/16-epoch selection rule and subsequent fresh full24 final training remain registered; no selection checkpoint is a final model.

Three failure sidecars were written underneath the detached mount. The mount supervisor refused to cover that nonempty directory. Those three owned files were moved together to a private preservation directory with byte hashes; no old result was deleted. The existing mount configuration was retained.

Running training writes local checkpoints and logs. A separate backup process publishes immutable archives under the NAS `durable_recovery_snapshots` subtree, then verifies the remote bytes. A pending copy is not a successful backup. Original NAS training artifacts remain intact. The completed TRAIN audits and B0 scenes1–20 retain their NAS locations. New scene predictions write local storage, with verified background NAS archiving.

NAS reads remain slow. Previously successful output-byte-verification receipts may be reused for scheduling while NAS is unavailable. This does not claim a new NAS recheck and never authorizes scoring. Explicit canonical scope adoption still rereads every prediction byte and strictly loads all three final full24 checkpoints before scoring; the inference-only reference authorization cannot run evaluators.

All 43 offline analysis checks passed (14 controller, 15 reference-scope, 10 identity-event, 4 backup). The separate frozen package suite remains 304 passed. This phase changes analysis/scheduling and documentation only, with no model, training rule, threshold, GT or frozen inference-source changes.

Current progress and verified backup archive hashes are recorded in [`v5_storage_recovery.json`](../results/v5_storage_recovery.json). Neither TRAIN loss nor these engineering checks demonstrate an improvement over GMT.

## Continued NAS disconnects and complete B0 reproduction

At 2026-10-26T21:53:02.754725+00:00, B0 has completed all22/29,019 paired times/548,583 candidates and exactly reproduced original C1 fields at every time. B1 is 17/22 at the recorded snapshot. This is GT-free inference; native scoring is still deferred until all three fresh final full24 gates.

NAS disconnected again at approximately21:34 and21:49 UTC. The local trainers continued with unchanged PIDs/optimizer/RNG. The reference scheduler exited on a real errno107 during manifest precheck; no reference model was running then. New predictions now use local directories; previously completed NAS scenes remain read-only links, and the previous NAS-direct root link is preserved. Successful prior receipts prevent rerunning completed scenes.

Remote writers now hold directory descriptors for the actual SSHFS mount, verify their mount IDs and revalidate the mount after remote readback. An unmount cannot redirect writes into the local underlying mount directory. Prediction archiving checks all original output bytes, never overwrites a differing NAS scene, preserves incomplete attempts, and reclaims local duplicates only after verification and model-worker exit. Atomic directory/symlink exchange preserves reader paths. Copy failures retain local originals.

B0 scenes1–20 were timed with NAS output; new B0 scenes21–22 and all other new predictions use local output. Cached wall FPS must not be interpreted as a controlled algorithm speed ranking. The registered real RGB timing still uses common local output storage for all methods. The 54 offline analysis checks passed; no model, registered training rule or GT changed.
