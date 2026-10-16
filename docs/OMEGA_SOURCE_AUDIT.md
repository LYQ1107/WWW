# VGGT-Ω 源码、历史证据与权重来源审计

本轮独立分支为 `research/vggt-omega-geometry-proof-20261016`，工作树为
`/data1/liuyeqiang/WWW_omega_geometry_20261016`。基于固定提交
`c6999a7ab1d22549816b5219734f53c8dcb74178`；代理恢复后实际检查远程，
参考分支仍为此提交，新 Ω 分支当时不存在。旧工作树、实验产物和 GT 保留。

已阅读原任务指定的八份历史报告，精确 SHA 见 `results/omega_preflight_v1.json`。
旧几何 screening 的420个变体接受0状态，重复变体不算独立样本；完整 dense run
14324次调用只有3次数值/背景 gate 通过，B5/B7 三个训练头最终输出均相同。
不能据此推断物理视野无重叠或把旧特征收益归为几何收益。
TrackVGGT 的56次教师实验未建立独立 VGGT 关系收益；原 val300 没有已认证的
原生实际主求解错误，教师排序改善不代表 Hungarian/Bank/最终 ID 纠错。

## 官方来源与实际阅读

- [官方仓库](https://github.com/facebookresearch/vggt-omega)，当前固定源码
  `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`，外部独立克隆保留。
- [论文](https://arxiv.org/abs/2605.15195)，2026-05-14 v1。
- [复现说明](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/reproduction.md)：
  benchmark 应使用416重训权重；原512仅为单独自然场景诊断。作者说明潜在数据污染风险，
  并未认定原权重实际发生污染；不得夸大这个限定。

实际阅读 `models/vggt_omega.py`、`utils/pose_enc.py`、`utils/load_fn.py`、
`models/heads/dense_head.py`；同时检查 camera head、rotation、attention、aggregator、
依赖和 LICENSE。完整源码文件 SHA 记录在 preflight。没有套用旧 VGGT API。

源码确认：patch16；balanced按416²附近 token预算计算实际长宽，非固定416×416；
极端宽高比先中心裁剪至[.5,2]，resize后形状不一致时白色居中padding；
RGBA在白底合成。pose编码是平移3、XYZW四元数4、垂直/水平FoV2；解码为
OpenCV camera-from-world `[3,4]`。depth为 `exp(logit)`，confidence为
`1+exp(logit)`，二者fp32；confidence不是概率。Camera/depth head保持fp32，
aggregator自动选择设备支持的BF16或FP16。requirements要求torch≥2.6。

## 权重访问与公开镜像核验

官方 [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega)
固定 revision `1041e80fc0e911235d3426b0a3d9a81075111a53`，API标记 `gated=manual`。
未认证416下载真实返回401。已检查标准缓存和凭据环境变量存在性，未打印或保存私人令牌。
状态 `BLOCKED_WEIGHT_ACCESS` 仅指当前416官方访问，不能写成模型几何负结果。

|官方权重|字节|官方LFS SHA256|
|---|---:|---|
|416 reproduce|4576703488|e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896|
|512|4576706117|c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934|

用户后续明确要求查找个人保存/替代来源及代理访问。已通过代理读取官方源码，
搜索文件精确名称、公开HF仓库、网盘线索与ModelScope。查询记录与公共API原始响应
保留在外部 `source_audit/`；没有发现可核验的416原始PyTorch公开镜像。
这只是本次有限搜索结论，不宣称全网不存在。

- `Asher-1/map_anything_vggt`：有416/512 GGUF转换文件，无原始PT；不作为主实验替代。
- `mlx-community/VGGT-Omega-1B-512-bf16`：MLX转换，非本机官方PyTorch接口，不采用。
- `ldn060904/vggt-omega`、`aoliong/vggt_omega`：只有仓库说明，无权重。
- `wincentIsMe/VGGT-Omega`：声明512原版并附原许可证，但容器SHA/大小与当前官方不同，
  没有凭声明直接认作字节一致权重。
- [1kaiser/vggt-omega-jax](https://huggingface.co/1kaiser/vggt-omega-jax/tree/a8c3a718e0cf78e9e4c6847229efea793d37f060)：
  公开512 `.pt` 的LFS SHA及大小与官方一致。已启动公开文件下载；必须实际下载后再做
  SHA验证及strict load。下载来源记录为第三方公开分发，绝不写成已通过官方HF账户授权。

模型/代码的官方FAIR Noncommercial Research License及其SHA已保留；只用于当前
非商业研究，后续分发仍须携带原许可。本轮不下载未知量化模型，不尝试读取受保护仓库
的隐藏文件或绕过认证端点。512下载及试验不能解除416主实验访问待办。

## 已固定的数据范围

复用12个已完成连续原生审计，逐个核对其manifest与原始产物SHA；未重跑GMT、未重新匹配GT。
总169468查询/1553认证错误中，650有任一同步对侧原生参考，314有同步纯净正例，
其中297的同步正例在Top8。903个错误没有同步对侧参考，第一版几何不可直接适用。
这些是输入适用性统计，尚无可靠几何或纠错证据。

fit11为154911查询/1527错误，其中629有同步对侧参考，304有同步纯净正例，287在Top8。
DEV12为14557查询/26错误，21有同步对侧参考，10有同步纯净正例，10在Top8；
23个全历史纯净正例不能等同于23个当前同步几何正例。两组明确分开。

固定cohort含66组GT无关均匀时间样本和30组TRAIN困难诊断，共96组；首24组
全部GT无关、覆盖11个fit场景。困难事件按场景/人物轮转，同一人物跨两个相机至少间隔30组。
索引、192张原图SHA、物理camera_id、原始实际尺寸均在manifest中。
GT标签保存在单独离线scope结果，模型输入manifest没有GT身份标签。

首次本地未提交索引按camera/person去重，源码检查后在任何Ω输出产生前修正为跨相机按person去重；
首次文件及日志已归档外部 `cohort_prefreeze_initial/`，正式v1只在修正后提交冻结。
未使用DEV5或TEST进行选样或调参。DEV12仅复用已有标签做适用性诊断，未加入96组fit图像。

## 后续实际执行状态

上面的“下载/加载待完成”描述首次preflight快照；后续512文件已完整下载并通过实际SHA核验，
在独立torch2.6.0+cu118环境完成1411项strict load及全部24组/48次AB/BA前向。
当前来源、结果与成本见 `OMEGA_PHASE0_INTERFACE_AUDIT.md`、
`OMEGA_FINAL_RESEARCH_REPORT.md` 和 `results/omega_numeric_evidence_v1.json`。
官方源码许可证与HF官方LICENSE的Git blob均为
`01465aec627c2698f972ca68abcabf54a497db14`，复制到本地权重目录；没有对第三方文件的许可重新命名。
416的官方访问仍然401/受限，主科研目标没有完成。
