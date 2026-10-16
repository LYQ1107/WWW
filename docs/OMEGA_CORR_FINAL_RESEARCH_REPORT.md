# Ω-Corr 最终研究报告

本轮完成160组、冻结Ω512/旧VGGT/DINO/原生GMT的全部注册独立机制对照，未达到独有跨视角证据GateB。按固定门槛停止新增训练、Ω-Corr solver插入和完整视频评测。结论只适用于该数据、层、局部匹配定义与有限预算，不否定一切Ω身份用途，也不援引旧Ω-F负结果作为证明。

- `TOKEN_CORRESPONDENCE_EVIDENCE`：`REAL_TOKEN_INTERFACE_PASS; DISCRIMINATIVE_ADVANTAGE_GATE_NOT_PASSED`
- `UNIQUE_OMEGA_EVIDENCE`：`NO_UNIQUE_OMEGA_CORRESPONDENCE_EVIDENCE`
- `NATIVE_GMT_CORRECTION`：`NOT_EXECUTED_GATE_B_FAILED`
- `FULL_TRACKING_EFFICIENCY`：`NOT_EXECUTED / NOT_MEASURED`

| 方法 | 查询 | 有效证据 | 正负可比行 | 正例分数更高 | 已知行修正 | 行破坏 | 净修正 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ω L11 F2 mutual | 384 | 164 | 122 | 50 | 1 | 94 | -93 |
| Ω L17 F2 mutual | 384 | 164 | 122 | 45 | 0 | 95 | -95 |
| Ω L23 F2 mutual | 384 | 164 | 122 | 46 | 1 | 92 | -91 |
| Ω L11独立单视图 | 384 | 164 | 122 | 50 | 1 | 94 | -93 |
| 旧VGGT L11 | 384 | 130 | 99 | 40 | 0 | 73 | -73 |
| DINO | 384 | 130 | 99 | 52 | 0 | 61 | -61 |
| Native1024纯外观 | 384 | 213 | 133 | 128 | 1 | 56 | -55 |
| NativeGMT全历史 | 384 | 213 | 133 | 130 | 0 | 0 | 0 |
| NativePrior原参数+2 | 384 | 213 | 133 | 130 | 0 | 0 | 0 |

仅Fit选择Layer11：净行修正{'11': -96, '17': -98, '23': -103}。机制检查有1次已知行修正、94次破坏，独有修正0。该强行视觉排序结果不能直接代表有界融合或实际ID效果。

| 同候选对照 | Ω减对照R1 | 95%场景/人物聚类区间 |
|---|---:|---|
| FULL_IMAGE_MULTIVIEW|omega|F3|11|M0|primary|0 | -0.010309278350515464 | [-0.08423940949935815, 0.04675122385402757] |
| FULL_IMAGE_MULTIVIEW|old_vggt|old_vggt|11|M0|primary|0 | 0.020618556701030927 | [-0.06897405257767156, 0.10231331168831166] |
| FULL_IMAGE_MULTIVIEW|dino|dino|23|M0|primary|0 | -0.07216494845360824 | [-0.22448979591836735, 0.0744172128811627] |
| NativeAppearance1024 | -0.4948453608247423 | [-0.7046042319749216, -0.3188405797101449] |
| NativePrior2 | -0.5257731958762887 | [-0.7272727272727273, -0.35714285714285715] |

研究问题逐项回答：

1. 联合视角是否优于单视图：共同候选聚类区间未建立稳健优势，不能声称独有跨视角交互收益。
2. 层价值：11/17/23全部完成，Fit选11；三个层均未跨过门槛，不按机制检查挑更好层。
3. 覆盖：1972物理proposal中916个（46.45%）少于2真实M0 patch；局部多点不可用，保留ROI/裁剪对照。
4. 人物与背景：M0/M1/M2均单列；同候选目标对背景环的区间见paired_controls.csv，不将背景共现当人物几何。
5. Mutual与ROI：raw均值和ROIAlign均完成，同候选表没有支撑本注册独有身份假设；不能靠不同可用分母宣布优越。
6. Cycle：软循环是附加数值，硬循环与mutual覆盖等价；没有满足门槛的额外真实身份纠正，实际求解未测。
7. 对旧VGGT/DINO：各层修正、破坏和净差见上表/完整matching报告，共同有效97条机制正负查询的区间见上。
8. 独有修正：Fit选层的共同候选、对强对照独有且修正原全历史Top1错误为0。其余固定层/cycle/local结果全部保留，不能事后择优提升为主结论。
9. 适用性：1895查询中992无同步跨相机参考，1383无可认证同步纯正例；原132主关联错误中103有同步纯正例。原完整TRAIN的169468/1553/650/314/297分母保留，不把有限样本当总体。
10. 已知修正/破坏：本报告数字为冻结行排序，原实际384机制查询有28主关联错误，但仅4条全历史Top1已知错误。二者不能混算。
11. GTA/Hungarian：没有新Ω-Corr状态分叉，变化数未测。OFF上下文重建用于严谨原生对照，不能冒充新证据进入求解。
12. 最终ID：NOT_EXECUTED，不能写0变化或0增益。
13. 调用预算与系统延迟：10%/20%不确定性在线触发未执行，系统FPS/P95未测；离线子进程和forward成本另见效率报告。
14. 为什么不直接NativePrior/DINO：本轮未提供需要更大Ω模型的证据。NativePrior原参数2/cap2与纯1024 GMT外观在同候选上是必须保留的强基线；本轮不证明它们完整视频一定提升。
15. WWW独立贡献差距：没有可复现的Ω独有跨视角净纠错，没有原生solver最终身份净收益，没有完整视频精度—在线成本证据；不能以工程实现充当论文贡献。

小目标补救已执行：固定32组224/336同像素crop，Ω/DINO各686次真实独立forward。机制子集73查询、26条有可认证共同正负；Ω没有稳健优于同crop DINO，更多编码patch不等于新增原始细节。所有主配置、安慰剂、失败及未知保留。

复现：使用final_audit内每个receipt记录的精确source_commit、command、环境和输入SHA，在新的输出目录执行对应脚本；不要覆盖原实验。冻结原始候选分数、共同候选索引、逐query输出已发布。独立CPU复算命令：`/data1/liuyeqiang/WWW_omega_runtime_20261016/venv/bin/python geo_tools/omega_corr_reproduce_rows.py --output /tmp/omega_corr_independent_rows.json`（新输出文件，禁止覆盖原结果）。README式细节见SOURCE_AUDIT、TOKEN_INTERFACE、DATA_AND_COVERAGE、MATCHING_RESULTS、CONTROL_EXPERIMENTS、ACTUAL_SOLVER_RESULTS、EFFICIENCY_REPORT、REVIEWER_AUDIT。
