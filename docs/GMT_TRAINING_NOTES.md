# GMT 训练运行记录

## Backbone compatibility

`checkpoints/backbone/CH_FPN_1x.pth` 是百度云原始文件，SHA256：

```text
c60ae4dc51882b230a39f9783121bbf6f31651c63341f4353c46e5fcda1053e6
```

仓库的 DFConv wrapper 需要两个 key 名称适配。`reproduction_tools/adapt_backbone_keys.py` 生成 `CH_FPN_1x_key_adapted.pth`，只移动以下 state-dict key，tensor 值逐项 bitwise unchanged：

```text
proposal_generator.centernet_head.bbox_tower.9.weight
    -> proposal_generator.centernet_head.bbox_tower.9.conv.weight
proposal_generator.centernet_head.bbox_tower.9.bias
    -> proposal_generator.centernet_head.bbox_tower.9.conv.bias
```

派生 checkpoint SHA256：

```text
5368aae363477bf1ad71c5f643ef3e776187c5b0effc1753ed4e6cace3ad9f20
```

直接载入原始 key 的训练 smoke 出现 NaN ReID loss；适配后单卡和分布式 smoke 的 loss 均 finite。因此正式 Stage1 使用派生文件，并保留原始文件不覆盖。

## 已完成的 smoke

- 单卡 mapper/model forward：PASS；`loss` finite。
- 8 GPU、canonical `TRAIN_LEN=10/ TRAIN_SIZE=1280`、activation checkpoint、20 iter：PASS。
- 8 GPU、短 clip smoke：PASS。
- 9 GPU canonical smoke 曾通过；经 1/2/4/9 卡实测，当前正式延续链固定采用 GPU0 单卡、`SOLVER.IMS_PER_BATCH=1`，因为该主机上的单卡路径迭代吞吐最快且最稳定。GPU1 的 12 GiB 外部残留上下文未使用。

正式 Stage1 运行目录：`outputs/stage1_single_gpu`，从原始 Stage1 的 `model_4000.pth` 延续到 local `model_16000.pth`（对应 global 20,000）。运行时使用 `GMT_CHECKPOINT_BACKBONE=1` 的 activation checkpoint、Gloo/CPU-staged collectives 和派生 backbone。当前 `model_13000.pth` 已验证最后 iteration、scheduler/global iteration、所有参数 finite、optimizer state 和 checkpoint reload；最终 checkpoint 仍必须再次验证后才允许启动 Stage2。

## 重要限制

当前正式延续链使用单卡 batch size 1，而不是原论文环境的 10-GPU global batch；这项硬件/吞吐差异必须进入最终 reproduction report，不能宣称 bit-for-bit 官方训练复现。GPU1 的外部 CUDA context 仍保持不动。
