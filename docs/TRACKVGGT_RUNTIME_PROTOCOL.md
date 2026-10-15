# Fresh native system and retrieval replay

Status: REGISTERED_AND_WAITING_FOR_NAMED_TEACHER_JOBS. No new runtime number is
claimed by this document. Source/commands are frozen in
manifests/trackvggt_runtime_scope_v1.json and trackvggt_live_baseline_v1.json.

Actual full image replay will compare frozen causal GMT OFF and the unchanged
head-free Native Prior at2/4/10 configured logical streams, using1/2/5 distinct
strict-development two-camera videos. Each stream starts at its real first frame
and ends at its original final frame. Configured10 streams therefore does not
mean a single synchronized10-camera scene or a steady10-active-stream load for
the whole unequal-length replay; active membership decreases as videos end.
Same-scene cameras use original frame synchronization; unrelated scenes use
ordinal interleaving. No TEST, GT query selection, teacher/student token cache,
new detector, or network/Internet request-latency claim.

Controllers wait for the named existing QD/cycle/native-control handles and
verify project GPUs0/4/5/6 are free. Then actual40-group OFF/Prior interleaved
prefixes must match the old complete native references exactly. Full runs cannot
proceed without that passing gate. The six full/two prefix producer processes
are bounded3500/120 seconds and1.25/.125GiB each, within the aggregate6GPU-wall-hour
and8GiB initial scope. CPU dependency waits and GT evaluators are separate.
The old raw data/checkpoints/results are not overwritten or deleted.

The actor/index are reused from run_live_retrieval.py. Passive mapping/step/query
wrappers report file-read+preprocessing, synchronized detector/GTA/Bank/tracker
steps and index query compute. Both methods charge identical profiling syncs.
The complete parent process wall time includes Python imports, model setup,
reads, replay, publication, response/log serialization, export and accounting.
CUDA allocator peaks are observed before legacy reset and after replay, so setup
peaks are retained; host peak RSS and independent sampled device/process data
are separate. These scopes do not pretend that query compute alone is full FPS.

The bounded100MiB clip index uses real detected native1152 vectors, clip30/TTL180
rounds and fixed query interval20. Every current target at query rounds is retained,
including GT-unmatched requests. All current arrivals publish before their query;
no later frame or encoding backfills an earlier snapshot. Other-camera clips and
unrelated-scene distractors share ranking. Offline GT labeling preserves actual
queries and uses the original one-to-one IoU/strict-majority clip convention,
with conditional Recall/MRR and all-request coverage reported separately.

Full native/Prior outputs are checked against the original standalone videos.
Derived per-video receipts bind exact prediction bytes and the common parent
manifest; they do not invent per-scene speed values. The10-stream complete
strict5 predictions then enter the existing TrackEval suite evaluator for HOTA,
DetA, AssA, IDF1, CVIDF1, CVMA, MOTA and IDSW; raw CLEAR output also retains Frag.
No alternative or repaired GT evaluator is introduced.

Six old native/Prior full-suite references (three actual inference seeds) have
been reverified by their raw-GT, prediction and run-manifest hashes. They remain
explicitly inherited baseline references, not new TrackVGGT teacher/student or
new timing experiments. Fresh replays and conditional student deployment remain
separate evidence. No deployed student or online VGGT-teacher FPS is manufactured
when there is no validated relation/student checkpoint.

The separate trackvggt_queue_stage_costs_v1 CSV/JSON records actual completed
queue producer wall time, including failed executed stages if any. Explicit
CPU-only evaluators/waits are excluded from GPU wall time, and running stages
remain pending. This is assigned-GPU process wall including startup, validation,
hashing and export, not active kernel time or a total-project ledger. Manual
probes, initial selected capture, CPU diagnostics, inherited work and the separate
DEV12 release ledger are disclosed exclusions. Passive monitor extrema and
overlapping observation windows are never summed as extra experiment runs.
