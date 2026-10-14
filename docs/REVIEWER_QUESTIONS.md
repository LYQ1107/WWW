# Reviewer questions — current evidence

This file is generated from current artifacts; pending items are not claims of completion.

## Q1: 为什么 RPCE 之外需要动态几何？

尚未证明需要。冻结相机几何在最小开发样本上未通过背景有效性门槛；传统几何对照和软平面分支已运行，不能据此声称新增动态几何有效。证据：geometry_validation.csv、vggt_geometry_validation.csv、plane_geometry_validation.csv。

## Q2: 为什么不直接使用 DINOv2/ReID？

必须保留强对照。12 个离线 GT 匹配困难查询上，GMT ReID Recall@1=100.00%，高于当前 VGGT 中间层；独立 DINO 的跟踪级对照正在补齐。不能用弱 DINO 对照替代现有 GMT 外观表示。证据：hard_feature_screening.csv、dinov2_screening.csv。

## Q3: VGGT 多层是否提供独立身份信息？

未证明。第17层 ROI 的256维与原始维度 Recall@1 分别为 50.00%、75.00%，表明随机压缩有明显损失，但 GMT ReID 在这12个查询上仍更好。VGGT修正的提案错误也能被GMT ReID修正，缺少独立互补证据。

## Q4: 为什么不简单拼接所有输出？

简单四层平均相似度等价于等权拼接归一化目标向量，已完成 TRAIN 标量校准并启动 B3 完整视频对照。该对照未结束，不能声称 GCIR 优于普通组合。

## Q5: 与 VGGT-Det 的差别？

当前实现做目标身份候选残差，而非3D检测；未复制其MMDetection3D/PointNet检测头。冻结多层聚合思想已有先例，任务不同本身不足以证明方法创新。见 GEOMETRY_OPEN_SOURCE_AUDIT.md。

## Q6: 与已有 VGGT+ReID+3D tracking 的差别？

待证明的区别是动态几何有效性、身份表示互补和因果计算分配。已有校准自由室内方法使用VGGT/OsNet/姿态提升和聚类，简单几何+ReID已是先例。三个新增假设尚未取得完整证据。

## Q7: 运动无人机如何保证几何一致性？

不比较不同调用的3D坐标。当前用单次联合调用相对相机关系，独立背景检查，失败即中性。合成相机投影和传播方向测试通过；航拍相机几何的实际有效性尚未通过。

## Q8: 少重叠视野是否有效？

没有此类有效性声明。无纹理、无足够可靠匹配、低覆盖、非有限或过期状态均为中性。ORB/SIFT和平面分支给出了实际低有效率，不强行拒绝身份候选。

## Q9: 身份触发优于固定周期吗？

未评测完成。独立调度器预算/前缀测试通过，但尚未接入实际同步/异步流水线；配置开启未支持模式会明确报错。纯margin采样选出了12个无GT匹配检测，显示身份需求代理可能浪费预算，需结合可靠性诊断。

## Q10: 相较 StreamVGGT/FastVGGT 的价值？

不能直接移用长序列加速倍率或流式权重。源码接口和许可证已审计；短双视角速度实测、公平预算对照和真实流式集成仍待完成。

## Q11: 完整 FPS/延迟/显存/参数量？

完整因果 B0 park18 实测已记录：当前图像读取到CPU身份输出，包括全部1029帧组；详见 runtime_results.csv 和 online_B0_full_v18 manifest。静态模型初始化排除并明确标识；VGGT冷启动算入输入延迟。完整新方法资源比较及参数量清单仍未完成。

## Q12: 提升是否来自不同检测/配置/评测 bug？

目前没有提升。基线固定checkpoint/config与映射修复，完整五段严格开发集：HOTA=81.0547、CVIDF1=95.6317。OFF、零系数、真实VGGT计算中性对照与整段采样预测一致性通过；B2前两系数首轮输出与B0一致。

## Q13: 三个模块能否分别证明有效？

尚未证明。当前 SCIENTIFIC_SUCCESS、ENGINEERING_COMPLETE、PAPER_READY 均未达成。B0/B1/B2 和几何诊断不能替代完整 B3–B9 消融、因果速度和重复复核。

## Q14: 为什么符合 WWW？

仍有实质适配风险。官方CFP要求明确Web科学问题，仅用网络数据集或API不充分。VisionTrack目前提供计算机视觉证据，真实多源内容检索任务和系统比较仍待实施。见 WWW_SCOPE_NOTE.md。

## Q15: 在线视频关联检索的真实贡献？

目前仅有离线目标检索诊断，没有可声称的Web系统贡献。真实跨片段查询、Recall@K/MRR、查询延迟、流数量/预算曲线和在线可用性需要继续完成。
