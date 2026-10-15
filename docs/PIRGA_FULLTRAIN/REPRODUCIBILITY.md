# PIRGA Phase 0 reproduction

Worktree `/data1/liuyeqiang/WWW_pirga_20261016`; runtime
`/data1/liuyeqiang/WWW_pirga_runtime_20261016`; Python
`/home/liuyeqiang/anaconda3/envs/GMT/bin/python`. Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
Existing result directories refuse overwrite. Replace output suffix only for an authorized repeat.

```bash
python geo_tools/audit_pirga_fulltrain.py --producer /data1/liuyeqiang/WWW_trackvggt_20261015 --runtime /data1/liuyeqiang/WWW_trackvggt_runtime --out NEW_AUDIT_DIRECTORY
python geo_tools/check_pirga_real_epoch_order.py --source-manifest results/PIRGA_FULLTRAIN/SOURCE_SHA_MANIFEST.json --runtime /data1/liuyeqiang/WWW_trackvggt_runtime --out NEW_DRY_RUN_DIRECTORY
python geo_tools/probe_pirga_indexed_store.py --pins results/cv_pil_v3_native_validation/P0_REAL_PREFIX40_MAPPING.json --root /data1/liuyeqiang/WWW_trackvggt_runtime/dev12_native_precheck_v1/on40_v1 --out NEW_CPU_REPACK_DIRECTORY
python -m unittest discover -s geo_tools -p 'test_pirga_*.py' -v
python -m unittest discover -s geo_tools -p 'test_cv_pil_*.py' -v
```

Fresh image prefix export is not part of these CPU commands. Its concrete one-shot command and
pending resource fields are in `results/PIRGA_FULLTRAIN/PREFIX_GPU_RESOURCE_REQUEST.json`.
It refuses a proposed permit. Do not edit that file to simulate approval. The separate actual
permit must preserve the authorization reply and old ledger references, then be SHA-pinned.
Full training/online stages are not enabled by a prefix PASS.

`pirga_native_trace.py` reuses original-repository passive trace code from frozen producer
commit `f97cf27f0f4299269c3bd1dd885a92e90639f51c`, source SHA in its header. Changes are explicit
video-ID output binding and local primitive import. Tracker core files retain their exact
original SHA. Fresh OFF/ON parity remains pending; no untested runtime result is asserted.
