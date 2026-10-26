# WorldMOT V4 C1 full TEST

Scene1完整工程预检通过：C0对存档预测及冻结R2原始代码的1005/1005时刻精确一致；C1只开启运动预测，6D状态、协方差和幂等检查通过，12,465个候选bbox/score/ROI差异为0。C1 Scene1输出复用于全量结果。

第一次CPU预检在第64时刻出现约1.4e-12的存档中心差异；查明原R2使用CUDA FP32后，恢复到同型号V100并完整复核通过。没有放宽数值比较，没有修改模型参数。失败尝试保留在预检JSON。

当前正在剩余21场并行因果推理，正式分数尚未测得。完成全部22场、29,019时刻、548,583个候选冻结后，才读取原始TEST GT进行14合法场景TrackEval、MATLAB和只读归因。所有旧结果不覆盖。

287项CPU测试通过，零失败、零跳过。完整固定协议见 [V4_C1_FULL_TEST_PREREGISTRATION.md](V4_C1_FULL_TEST_PREREGISTRATION.md)。
