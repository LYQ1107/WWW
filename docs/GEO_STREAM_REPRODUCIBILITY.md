# VGGT+GMT reproduction and retained evidence

This branch is independent of JEV. The committed split, frozen GMT checkpoint and original baseline lock define the experiment. `results/baseline_manifest.json` records historical official TEST exposure; no current model selection uses TEST. Development is unseen by the new association heads, not by the pretrained GMT backbone. Seven original development scenes have conflicting raw GT identities and remain separate diagnostics; the five strict scenes are never replaced by those diagnostic metrics.

## Check the retained inputs without changing them

Use the project environment from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python \
  geo_tools/preflight.py
```

The current entry reads the committed split and baseline lock, checks the current GMT checkpoint, YAML, TRAIN annotation, retained raw-GT files, original offline suite and prediction hashes, and writes a new report under the external runtime `preflight/` directory. It does not regenerate the split or replace the baseline manifest. `--report-output` accepts a new filename and refuses an existing one. A metadata-only PASS does not claim another model load or inference. `--build-model` additionally performs the existing strict GMT state load on an allocated CUDA device, reports the loaded state-key count, and still runs no video inference. The historical initializer is available in Git and is not the current reproduction entry.

Data and large weights stay outside Git. This workspace uses `/data/DATASETS/TRACKING/JDE/VisionTrack`, the checkpoint path captured in the baseline lock, and `/data1/liuyeqiang/WWW_vggt_geo_runtime` for the virtual environment, pinned external sources, weights, heads, raw outputs and logs. See `GEOMETRY_OPEN_SOURCE_AUDIT.md` for external sources and licensing scope.

## Reproduce inference from fixed source and parameters

The JSON plans in `manifests/` contain exact argv arrays, dependencies and GPU assignments. Source manifests ending in `_sources.json` identify the prediction files used by each native suite. The runtime `queues/*/manifest.json` records the executed stage command, source provenance, child PID, log path, exit code and log SHA; `FINISHED` only means the declared commands ended. Diagnostic failures and superseded unstarted controllers are retained.

Create an immutable Git worktree at the selected source commit before inference. Use a new run tag and matching source manifest for a fresh run; existing output tags refuse overwrite. Changing output names does not authorize changing the split, coefficients, head, inference RNG seed or protocol. Run serially through a declared queue on the project allocation, respecting the existing live queues and four-GPU limit. Do not edit pinned source or restart a live inference because a status observation timed out.

The relevant entry points are:

| Entry | Behavior and scope |
|---|---|
| `geo_tools/run_baseline.py` | Original offline GMT and explicitly selected offline ablations; full-video postprocessing retained |
| `geo_tools/run_online.py` | Separate causal GMT baseline/ablations, current/past input and externally available identities |
| `geo_tools/run_seeded_online.py` | Explicit actual model/stream inference RNG; fixed trained head, separately from TRAIN-head seed |
| `geo_tools/run_native_prior.py` | Fixed head-free control; inference seed is independent of TRAIN-head seed |
| `geo_tools/run_live_retrieval.py` | Actual image-to-tracking-to-clip-index/query producer; GT only in later evaluation |
| `geo_tools/run_decision_trace.py` | Diagnostic actual solver provenance; prediction gate before GT labeling; excluded from speed claims |
| `geo_tools/validate_physical_camera_input.py` | Explicit physical-ID input contract against retained order-failure fixtures; does not make internal query order invariant |
| `geo_tools/test_constructed_geometry_degeneracy.py --report-output NEW_PATH` | CPU procedural blank/unrelated-background and60-degree in-plane-rotation checks; synthetic detections/past F, exact invalid-geometry score fallback, temporal rejection and observed motion-trigger response; refuses an existing report |
| `geo_tools/audit_process_wall_runtime.py` | Actual complete queue launch/wait costs, sealed log/run/output hashes; excludes recorded held children and conservative priority-owner overlap; no zero-filled setup estimate |
| `geo_tools/monitor_project_resources.py` | Passive samples on allocated0/4/5/6 for explicitly named live project controllers/descendants; unique tag, per-PID RSS and NVIDIA process memory, whole-device utilization; exits after those jobs finish |
| `geo_tools/evaluate_predictions.py` | Native per-scene evaluation using the paired run manifest |
| `geo_tools/evaluate_suite.py` | Native TrackEval combination over every declared strict scene; does not average normalized per-scene scores |

Frozen GMT internals and its sequential camera update order are retained. The original whole-GMT swap diagnostic found different identity partitions for both OFF and B7. `PhysicalCameraGMT` is an explicit optional input adapter: sort by physical `view_id`, run the unchanged tracker, and restore outputs to caller positions. Its validation is distinct from the earlier VGGT encoder-only reversal test and from internal tracker invariance. Current dataset producers already supply canonical physical camera order. A partial adapter test cannot establish full-video or valid-pose invariance.

## Refresh derived reports after a declared run finishes

Copy only complete native suite metrics into `results/`, retaining their original provenance. `collect_native_suites.py` verifies the exact source specification, prediction/run-manifest/raw-GT hashes, scene scope and protocol before importing an existing completed report. With `--generate-ready`, it may combine only the two declared root-owned fixed-primary-head inference-RNG suites, after every scene has a successful full-video native metric. Original TRAIN-head queue combination stages retain ownership. Existing result files are preserved; incomplete suite directories or changed artifacts require inspection and are never silently retried. The analysis commands below read actual outputs; they do not perform inference or train/select a new model:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/collect_native_suites.py --generate-ready
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/audit_current_default_source.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/audit_suite_settings.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/audit_dense_geometry_ablation.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/audit_native_prior_stability.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/audit_identity_matrix.py --generate-missing
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/audit_dense_seed_stability.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/audit_process_wall_runtime.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/update_www_scope.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/summarize_results.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/update_reviewer_questions.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/plot_research_results.py
```

These derived report files are deliberately refreshed. Locked inputs and raw inference artifacts are retained. Missing complete suites stay pending. The identity matrix requires matching metric, prediction, protocol and raw-GT hashes; it counts unordered common GT-matched output relations, with additions/removals excluded and reported separately. Those counts are not actual Hungarian decisions. The separate decision-trace evaluator retains unknown queries, mixed reference tracks and averaged-descriptor memory limitations.

`results/experiment_manifest.json` indexes retained run manifests and their SHA values; each original manifest contains its recorded command, commit, environment, model/data hashes, cache and causal scope. `results/suite_settings_audit_v1.json` checks complete-suite active settings against their captured Git defaults. `results/plots/manifest.json` hashes every figure input and output. Missing historical per-run loaded foundation-weight/calibration hashes are a recorded provenance limit; current files are not silently presented as hashes captured by old runs.

Use `GEO_STREAM_REQUIREMENT_AUDIT.md`, the current raw queue/process state and all explicitly pending reports before deciding completion. Filename existence, a queue exit, a metadata check or a local replay API does not establish scientific stability, real-time operation, Web deployment or paper readiness.

Per-query outcomes can be written separately with `evaluate_live_retrieval.py --verify-existing --outcome-output NEW_PATH`; old metrics are recomputed and compared without replacement. Only the recognized early four-counter schema permits `--verify-legacy-counter-schema`, which still compares every retained number and rejects unknown legacy fields. Use `audit_retrieval_outcomes.py` to retain query corrections, harms, eligibility drift and common-cohort MRR; equal query/eligible counts or aggregate scores are insufficient. Sidecars are offline GT analysis and have no throughput meaning.

Final passive-resource aggregation uses `geo_tools/summarize_resource_samples.py` only after the named probe reports `FINISHED_EXISTING_JOBS`; it rejects live/failed probes before writing a final report. It verifies sealed CSV hashes and complete four-device sample groups, keys processes by PID plus start ticks/tag/role, and preserves per-GPU NVML samples separately from per-PID RSS. Its sample maximum is not a lifetime peak. First-sample presence flags possible partial coverage; later appearance is not proof of full process coverage. No device or process maxima are summed as total physical memory; sampled utilization means are explicitly unweighted. Two guards cover rejection of live/incomplete probes and duplicate GPU rows of a single process. Both probes have now ended: the original report has5709 sample groups and the balanced-handoff report1897, retained separately.

Complete feature-versus-joint geometry attribution can be repeated with `geo_tools/audit_dense_geometry_ablation.py --head-seed 20261014` (or the predeclared20261015/20261016 after that head has both complete five-scene predictions). The primary output filename remains unchanged; other heads use separate seed-qualified result filenames. Every report checks the actual trained-head checkpoint hash and fixed actual inference seed20261014, in addition to all existing paired model/data/options/call/output checks. A report for one head is not three-head attribution; missing inputs fail rather than substitute primary or partial scenes.

Balanced scheduling keeps the immutable inference source7b1c9da1 and every original remaining command. `manifests/primary_vggt_inference_balanced_handoff_v1.json` moves only unlaunched seed16 scene19/18 to freed project GPU4; replacement GPU0 waits for the existing seed16 scene10 child, evaluates it, then runs seed15 scene1. Actual source/passed-prefix hashes, live dispatcher/child identities, launched controller PIDs and the original queue snapshot are retained in runtime `handoffs/primary_vggt_inference_balanced_handoff_v1/manifest.json`. The original held dispatcher is superseded only after its child exits, so its last raw queue row is not relabeled PASS/complete. Its outer duration is explicitly unavailable even though the child's model outputs remain valid.

Both passive probes remain distinct: `resource_samples/remaining_declared_jobs_v1` and `resource_samples/remaining_declared_jobs_balanced_handoff_v1`. The first can mark replacement roots unattributed; that does not identify an unrelated user's workload. The second follows the replacement controllers and existing named roots. After each probe independently reports FINISHED_EXISTING_JOBS, use `summarize_resource_samples.py --samples PATH --report-output DISTINCT_RESULT.json`; a different probe cannot replace an existing report's provenance. Overlapping device/process samples are not independent repetitions or additive physical memory. Final coverage is assessed against both probe time windows and the handoff record.

`geo_tools/report_storage_budget.py` reads all runtime-root entries with GNU du apparent bytes and generates storage_budget_v1.json / GEO_STREAM_STORAGE_BUDGET.md. Frozen weights, runtime-local environment, immutable sources, compact features/heads/samples, retained prediction/retrieval/evaluation artifacts and controls/logs are separate. Symlink targets and external original DATA/checkpoint/environment storage are excluded. No cleanup is performed. Run it again after live writers finish for the final footprint; current figures are a time-stamped snapshot rather than a peak or atomic filesystem capture.

Actual-solver analysis uses the strengthened `audit_decision_trace.py`: both standalone prediction/default-configuration gates, frozen GMT/data/inference seed, exact proposal-source-to-public-detection mapping and common primary-query GT labels must agree. Source-index equality or equal query counts alone are insufficient. New cohort-verified reports use distinct filenames; original queue-generated v1 reports remain retained. GT labels enter only this post-run analysis; mixed/unknown references and averaged memory descriptors retain their original restricted scope.

## Final scoped audit and regeneration

The retained matrix is complete:42 native strict5 suites,35 paired output comparisons, all three declared dense seed factors and12 retrieval comparisons. Both resource probes are sealed. After the derived inputs and figures are current, run:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=geo_tools \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python -m unittest geo_tools.test_finalization_guards
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=geo_tools \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/finalize_research.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=geo_tools \
  /data1/liuyeqiang/WWW_vggt_geo_runtime/venv/bin/python geo_tools/update_reviewer_questions.py
```

Finalization verifies raw artifact hashes, input locks, complete cohort/protocol/seed gates, seven raw-GT diagnostic scenes, both sealed resource probes, source files, retained tests and all22 figure outputs. It writes `final_candidate_selection_v1.json`, `final_completion_audit_v1.json`, the requirement audit and the final report. Existing external test logs are evidence checked by the finalizer; generating a report does not itself rerun inference or the139 core tests. A repeated `summarize_results.py` checks the final audit's evidence hashes and restores the completed report/status; changed evidence fails explicitly and requires a fresh final audit. Recreating figures changes PDF timestamps/hashes and therefore also requires that refresh. The final audit covers scoped engineering deliverables and negative findings; stable VGGT-specific improvement and PAPER_READY remain unsupported.
