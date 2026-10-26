# V4 C1 — one authorized full TEST

Parent: `44b59bc1198ab069f64023e7c8d1339e01077029`.
Branch: `research/worldmot-v4-c1-fulltest-20261026`.
The human explicitly authorized one exploratory complete C1 TEST on 2026-10-26.
This supersedes the previous V4 experiment's prohibition on starting this TEST;
it does not authorize training, another model, parameter search or altered GT.

C1 is exactly `state_configs(1024)["C1"]`: only motion prediction enabled
relative to C0. Keep LEGACY_HARD, Joint Assignment, Joint Birth, current GMT
ROI and every timestamp's existing Ω geometry. Q/R priors, thresholds, lifecycle,
checkpoint and observation handling remain unchanged. All existing algorithm
files from the parent are preserved; the new entry point adds scope validation
and transparent numerical/timing observers.

Frozen Full weight SHA:
`cfda40050e89b31dd4480ca4d5010e23067a94ba18ea7d0e2425d5935d3f7fb8`.
Use `load_frozen_r2_head`, never disable the old strict checkpoint/scope loader.
The original full R2 scope is pinned to
`0fbb52166edf1c53e422e80707436787926893290842819aa378043c385f7dc6`;
its TRAIN scope is `f1160f7122dcc4e33570dc5c415ec7ddc46072982b287d4eade01858c7a13a74`.
The new TEST scope separately binds the current inference fingerprint, the
original locks, all input files and the human authorization. Private approval
records are saved with the run, not distributed as repository defaults.

First run the whole 1,005-time Scene1 C0 and compare archived R2 common fields,
and every numerical object state/head input against the original b448af4 core.
Added metadata is excluded, changed shared values are forbidden. Only after
exactness passes run C1 Scene1, preserving it as official output for the full
run. Check finite six-dimensional states/PSD covariance and zero-dt prediction
idempotence; no original annotation or raw GT file may be opened by inference.

Full coverage is 22 scenes, 29,019 paired times, 58,038 original image inputs
and exactly 548,583 frozen candidates. Check every key/bbox/score and the
unchanged ROI input signature. State resets occur at scene origins only.
Each scene has one nonblocking exclusive writer lock; completed SHA-verified
scenes are reused. Partial attempts remain explicit and are never appended
across a restarted tracking history or substituted for completed scenes.

Inputs use the previously byte-verified local mirror, with all original SHA
bindings rechecked. Scene outputs are staged locally, frozen and verified,
then archived by complete scene under
`/data1/liuyeqiang/nas/A100/WorldMOT/worldobject_v4_c1_fulltest_20261026/`.
Archive time is separate from cached inference time. No new Ω/detector cache
is extracted and no data/weights are downloaded.

Hardware choice uses engineering timing only. Default up to eight CPU scene
workers, reduced if memory/CPU availability requires. After the Scene1 gate,
compare one 32-time C1 CPU/GPU prefix if four authorized V100s are genuinely
idle. Ignore the first three prefix times for hardware warmup; choose four GPU
workers only if projected four-GPU throughput exceeds the available CPU workers
by 25% and the profile preserves candidate/ID semantics. No GT or identity
metric selects hardware. Otherwise use CPU. Never stop shared-server tasks.

After all22 predictions and SHA manifests freeze, run unchanged TrackEval and
VisionTrack MATLAB R2020a on raw-GT-legal scenes
1/2/3/4/6/7/8/9/10/11/15/16/18/19. The other eight scenes keep complete
predictions but never enter official scores through cleaned GT. Retain native
tool hashes, raw logs, count-based combined scores and per-scene results.
TrackEval IDSW and MATLAB CLEAR switches are distinct.

Read-only posthoc attribution extends the existing V3 analyses after freeze:
same-person splits, different-person merges, temporal changes, wrong reuse,
false Birth/NULL proxies, gap recovery, cumulative mixed IDs and one-to-one
retention. Report both corrections and regressions, legal14 separately from
blocked8 clean-subset diagnostics. Never use this analysis to rerun a tuned C1.

This is exploratory TEST: V4 design already saw V3 error analyses on the same
TEST. It is not a claim of first untouched holdout validation. Finish every
normal scene even if results are negative. No new training, new seeds or
additional C2–C5 TEST runs follow automatically.
