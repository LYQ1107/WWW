# 长任务监控约定

`reproduction_tools/monitor_research.py` 是当前 GMT/JEV 完整科研链的保守监控器。它每 60 秒更新：

```text
outputs/research_monitor/status.json
```

监控器识别固定的 Stage1/Stage2 命令以及 `chain_research_after_stage2.py`。若进程意外消失且对应最终 checkpoint 尚不存在，则最多从 `last_checkpoint` 原地恢复 3 次；Stage2 完成后，它会继续监控推理、counterfactual、baseline/JEV、Oracle 离线审计和 TrackEval/CV 评测，直到 `outputs/research_pipeline/PIPELINE_COMPLETE.json` 出现。每个后处理步骤都有独立 manifest 和断点产物；不完整产物会被保留为 `.incomplete.*`，不会静默覆盖。

它不会修改学习率、batch、GPU 数量、配置或 checkpoint，也不会在已有训练/后处理进程存在时启动第二个同类进程。Stage2 首次启动仍由 `chain_stage2_single_gpu.sh` 负责；若 supervisor 已退出而 Stage2 尚未完成，monitor 才会接管并用同一命令续跑。后处理 supervisor 同样由 monitor 自动启动和重启。

后处理 supervisor 启动时还会强制校验
`outputs/stage2_single_gpu/model_20000.pth`：iteration、scheduler/global
iteration 必须均为 20000，optimizer state 必须存在，参数必须 finite 且
checkpoint 必须可 reload；校验不通过时不会启动正式推理。

启动方式：

```bash
setsid /home/liuyeqiang/anaconda3/envs/GMT/bin/python \
  reproduction_tools/monitor_research.py \
  > outputs/research_monitor/monitor.log 2>&1 < /dev/null &
```

查看状态：

```bash
python -m json.tool outputs/research_monitor/status.json
python -m json.tool outputs/research_pipeline/pipeline_status.json
```

Oracle 只在离线 counterfactual replay 中使用，并在 manifest 中标记
`future_gt_access: true`；它不会被传入 online JEV runtime。

Stage2 完成后，`chain_research_after_stage2.py` 会并行启动冻结 checkpoint
的 train/test traced-OFF 推理（GPU2/GPU3），并在 GPU5 启动 JEV 完全关闭的
native-OFF 推理，用于严格比较预测 JSON；两者不等价时流水线会失败。策略训练
使用相互隔离的 CPU 进程并行执行；实际 JEV test inference 在 GPU4 后台执行，
与 CPU replay/evaluation 重叠。上述并行只改变调度，不改变 checkpoint、输入、
决策动作空间或评测定义。
