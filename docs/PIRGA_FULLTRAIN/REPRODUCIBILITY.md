# PIRGA reproduction and evidence boundaries

Worktree: `/data1/liuyeqiang/WWW_pirga_20261016`; runtime: `/data1/liuyeqiang/WWW_pirga_runtime_20261016`; Python: `/home/liuyeqiang/anaconda3/envs/GMT/bin/python`.

Phase0 is complete: all11 captures and independent admission passed. The historical resource request files describe earlier plans; actual allocations, consumption and the later conservative correction govern resources. Existing GPU permits are consumed, not reusable grants. Do not restart exports or GPU fitting from these instructions.

For current state, use:

```bash
cd /data1/liuyeqiang/WWW_pirga_20261016
CUDA_VISIBLE_DEVICES='' /home/liuyeqiang/anaconda3/envs/GMT/bin/python geo_tools/pirga_status.py
```

Frozen scientific protocols, source-code hashes, data paths and input SHA are in `results/PIRGA_FULLTRAIN/PHASE1_MATCHED_PROTOCOL.json`, `PHASE2_MATCHED_PROTOCOL.json`, `ORDINARY_DECISION_MATCHED_PROTOCOL.json`, and `EXHAUSTIVE_MARGIN_PROTOCOL.json`. Every fit verifies its source hashes before updates. Never edit their dependency modules while fits or fixed evaluation queues are active.

All common comparisons use113415 unique queries per epoch, fixed seed20261015, no replacement, endpoints1/3/5. The exhaustive-margin supplement uses113453 and567265 updates at5epochs rather than567075; it is explicitly separate. Native perception capture seed20261014 is distinct from head initialization seed20261015. No officialTEST or24-sceneCV run is part of these results.

A new-output CPU reproduction is allowed by the protocol; for example:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/liuyeqiang/anaconda3/envs/GMT/bin/python geo_tools/train_pirga_innovation.py \
 --protocol /data1/liuyeqiang/WWW_pirga_runtime_20261016/PHASE2_MATCHED_PROTOCOL.json \
 --protocol-sha 17ffe01425aa3da9e321c2b96d55eb30d82dccecfebdfadbff724f4563a552c6 \
 --out NEW_UNIQUE_CPU_REPRO_DIRECTORY
```

Do not overwrite existing outputs. Full CPU fitting is real work, not a lightweight validation command. Weights remain on the server with SHA in each CHECKPOINTS.json; GitHub contains only code, protocols and derived results. `--pins` on the replay executors requires an actual complete fixed-epoch checkpoint with matching source, seed, step and admission; engineering prefixes and fake step300 metadata are refused.

Phase1 used model-wise independent heads onGPU, then an exact completed-step joint AdamW checkpoint onCPU when the corrected existing quota bound was reached. All three arms move together. Floating-point execution betweenGPU andCPU or independent/grouped GEMM is not claimed bitwise identical; real TRAIN shadow tolerances are recorded. The original1.4h supervisor reports an expected terminal-state error because the correcting supervisor initiated an earlier stop; actual CPU progress is in `fit/STATUS.json` and `CORRECTIVE_RESOURCE_LEDGER.json`.

The GPU-stop resume was rolling and was consumed before later CPU snapshots replaced it. Its historical SHA is recorded, but an immutable copy of those original stop bytes was not retained. An immutable joint CPU checkpoint atstep500000, including AdamW and exact counters, is archived separately in `PHASE1_IMMUTABLE_CPU_RESUME.json`. Do not present that later archive as the original GPU-stop artifact. Fixed epoch1/3/5 model weights are retained separately. A resumed epoch's raw `epoch_wall_seconds` measures its resumed suffix; aggregate process wall time is in resource ledgers, not inferred by summing that field.

Current contract commands:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/liuyeqiang/anaconda3/envs/GMT/bin/python -m unittest discover -s geo_tools -p 'test_pirga_*.py' -v
```

Train applicability can be independently reproduced without video or rawGT rematching using `audit_pirga_native_margin_scope.py --protocol .../ORDINARY_DECISION_MATCHED_PROTOCOL.json --out NEW_SCOPE_AUDIT.json`; it must reproduce38 extra pure-positive queries,4 with controllable strict-past positive support. It does not modify the frozen common fits.

All reported development results are complete exposedTRAIN12 original-state primary Hungarian replays. Bank/Gallery/birth counterfactuals and full-imageFPS require separate admitted native online runs. GT is joined only after scores/actions for reporting; arbitrary interval witnesses use GT offline and are not learned repairs. The uncertainty inference-gate ablation is a same-checkpoint diagnostic, not another trained model: epoch1 is explicitly posthoc, with the audit fixed before later3/5results. No threshold, bound, horizon or model endpoint may be tuned from these diagnostics.

The separately frozen `LEGACY_CANONICAL_COVERAGE_PROTOCOL.json` tests the historical Top8/latest4 TRAIN interface. All3000 old actor inputs and positive aliases match, but19 old training queries have reduced negative-certification masks; strict equality failures remain published. A new canonical-mask300-step baseline is therefore paired with full106800-query epochs1/3/5 under identical conservative masks. Original old300 weights/results remain untouched. Historical preprocessing selects latest4 before removing same-frame entries, not after. Full DEV replay always retains complete native candidates/history. `publish_pirga_legacy_coverage.py` namespaces old300(S0), canonical300(S1), legacy-full(L1) and native-full(L2); cross-family budgets differ by design. Full L1 has534000updates/arm, not567075. No family is pooled as independent evidence.
