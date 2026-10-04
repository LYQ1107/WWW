# JEV runtime adapter

`gtr/modeling/jev_runtime.py` 定义了 online-safe 的模式边界：

- `off`：只返回原 GMT action；
- `shadow`：运行 JEV、记录 proposed action，但提交原 GMT action；
- `jev`：提交 JEV action；
- `oracle`：在 runtime 直接拒绝，Oracle 只能在离线 counterfactual labeler 中运行。

`JEVRuntimePolicy` 不持有 `track_ids`、`id_count`、`id_reid_dict` 或 memory bank，也没有 commit 副作用。只有 `JEVCommitAdapter.commit(...)` 接收显式 callback 后才可以改变 tracker state。这为 OFF/SHADOW 逐帧等价检查、未来 GT 禁止导入和 action permutation stress test 提供了清晰边界。

GMT 的实际接入点是 `gtr/modeling/meta_arch/gtr_rcnn.py` 中的
`run_first_tracker_plus`、`run_global_tracker_plus` 和
`run_memory_tracker`：association transformer 仍只负责 proposal/Hungarian，JEV
只决定 typed action，ID/memory 的更新仍在 GMT commit 代码中完成。默认
`MODEL.JEV.ENABLED=False`，因此原始训练和推理路径不加载 JEV。

显式运行模式：

```text
MODEL.JEV.ENABLED=True MODEL.JEV.MODE=off
MODEL.JEV.ENABLED=True MODEL.JEV.MODE=shadow MODEL.JEV.CONTROLLER_WEIGHTS=...
MODEL.JEV.ENABLED=True MODEL.JEV.MODE=jev MODEL.JEV.CONTROLLER_WEIGHTS=...
```

`off + TRACE_PATH` 可记录 frozen GMT 的在线 state/proposal，供离线 labeler 使用；
`shadow` 只记录 JEV proposal，提交的仍是 GMT action；`jev` 才改变提交结果。
