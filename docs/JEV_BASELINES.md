# JEV baseline contract

本文件冻结决策层比较时的输入和输出接口。所有 baseline 都接收同一份在线状态特征、typed question 和运行时合法 action mask；它们只返回 action logits/probabilities，不读取 GT，也不创建或修改 GMT 的 track/memory state。

## 特征前缀

为了让传统阈值基线和学习型基线在同一输入上比较，状态向量的前四列保留如下含义：

```text
0  current association / ACCEPT_CURRENT score
1  reassociation score
2  WRITE_MEMORY score
3  REACTIVATE_OLD score
```

后续列可以放 detector、ReID、轨迹年龄、时间/视角、margin、memory 和 disagreement 等状态特征。正式 adapter 必须在 manifest 中记录 feature schema 和 checkpoint hash。

## 已实现的比较对象

- `FixedThresholdPolicy`：GMT 风格固定阈值规则；不学习参数。
- `GlobalLearnedThreshold`：所有 typed question 共用一个可学习标量阈值。
- `StateConditionedThreshold`：阈值由当前状态条件化。
- `LogisticGate`：固定六个 semantic action slot 的线性 gate。
- `IndependentMLPHeads`：MATCH/MEMORY/REACTIVATION 三个独立 MLP。
- `SharedEncoderSeparateHeads`：共享状态编码器、三个独立 action head。
- `FixedSlotMLP`：一个 MLP 输出固定 action slots，再执行 runtime mask。
- `JEVDecisionController`：共享 state/question/action 编码器，可选无位置 option interaction；这是 proposed typed action-conditioned controller。

`FixedThresholdPolicy` 的正式默认阈值为 VisionTrack GMT 配置中的
`OVERLAP_THRESH=0.2`。它与 GMT 原始 OFF 路径分别报告，避免把原始
Hungarian + threshold 与允许 typed `REASSOCIATE` 的强 threshold baseline 混为一谈。

所有输出都按 action 名称 gather 后再 mask，因此交换 legal action 的输入顺序只会交换概率顺序，不会改变语义分数。`REACTIVATE_OLD` 等非法 action 不会成为死类别。

## 训练比较规则

训练脚本接收冻结 GMT decision trace 的离线 frozen-evidence mutable-state rollout；正式报告仍必须固定 detector、Stage1/Stage2 checkpoint、proposal、窗口、memory 配置、特征和 split，只改变 decision policy，并明确说明未重新前向 detector/association、也未序列化完整 Detectron2 `Instances`。Oracle 标签只能在离线 counterfactual builder 中生成，strict online/shadow/eval 过程不允许导入 future-GT 代码。

CPU 不变量测试：

```bash
PYTHONPATH=/data1/liuyeqiang/WWW:/data1/liuyeqiang/WWW/third_party/CenterNet2:/data1/liuyeqiang/WWW/reproduction_tools \
  /home/liuyeqiang/anaconda3/envs/GMT/bin/python reproduction_tools/test_jev_baselines.py
```
