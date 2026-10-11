# Phase XXI — Visual execution numerical parity

复用已授权的12个固定teacher图像/问题组、既有Qwen3-VL4B与公布QA LoRA及独立visual_env。原FP16结果22路径未通过rawlogit门槛但决策相同；用固定FP16默认、FP16math、FP32math三个执行profile区分backend与精度，门槛不变。选项顺序敏感性单独记录，不能把它混同KV错误。没有新模型、任务训练、native动作或新数据。最多1GPU、1800秒、新runtime64MiB、home≥30GiB。只解决执行一致性，不能由此宣称身份收益或解封训练。新数据来源问题仍待用户回复。
