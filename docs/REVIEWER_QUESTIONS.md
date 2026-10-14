# Reviewer questions — current evidence

This file is generated from current artifacts; pending items are not claims of completion.

## Q1: 为什么 RPCE 之外需要动态几何？

尚未证明需要。冻结相机几何在最小开发样本上未通过背景有效性门槛；传统几何对照和软平面分支已运行，不能据此声称新增动态几何有效。证据：geometry_validation.csv、vggt_geometry_validation.csv、plane_geometry_validation.csv。

## Q2: 为什么不直接使用 DINOv2/ReID？

必须保留强对照。12 个离线 GT 匹配困难查询上，GMT ReID Recall@1=100.00%，高于当前 VGGT 中间层。独立DINO稀疏/逐帧完整严格开发对照以及容量匹配GMT/DINO学习头主种子对照已完成，VGGT学习头未优于它们。证据：B1_dinov2_dense_strict_suite_v1_metrics.json、C_lowrank_gmt_seed14_k10_strict_v1_suite_metrics.json、C_lowrank_dinov2_seed14_k10_strict_v1_suite_metrics.json。

## Q3: VGGT 多层是否提供独立身份信息？

未证明稳定或VGGT独有增益。第17层 ROI 的256维与原始维度 Recall@1 分别为 50.00%、75.00%。另在45个固定TRAIN查询中，原维度VGGT修正1个GMT ReID/提案错误，但破坏20个原正确ReID查询；同一修正也可由独立DINO完成。有限互补允许小头原型验证，不能宣称几何身份信息独占。

## Q4: 为什么不简单拼接所有输出？

简单四层平均相似度等价于等权拼接归一化目标向量；B3完整严格开发输出与B0完全一致。B4标量weight2完整对照略低于B0。主种子低秩GCIR同样未建立收益。小头保留32维/两顺序64维目标表示，减少新增向量体积，但原GMT记忆和完整系统成本须分别测量。

## Q5: 与 VGGT-Det 的差别？

当前实现做目标身份候选残差，而非3D检测；未复制其MMDetection3D/PointNet检测头。冻结多层聚合思想已有先例，任务不同本身不足以证明方法创新。见 GEOMETRY_OPEN_SOURCE_AUDIT.md。

## Q6: 与已有 VGGT+ReID+3D tracking 的差别？

待证明的区别是动态几何有效性、身份表示互补和因果计算分配。已有校准自由室内方法使用VGGT/OsNet/姿态提升和聚类，简单几何+ReID已是先例。三个新增假设尚未取得完整证据。

## Q7: 运动无人机如何保证几何一致性？

不比较不同调用的3D坐标。当前用单次联合调用相对相机关系，独立背景检查，失败即中性。合成相机投影和传播方向测试通过；航拍相机几何的实际有效性尚未通过。

## Q8: 少重叠视野是否有效？

没有此类有效性声明。无纹理、无足够可靠匹配、低覆盖、非有限或过期状态均为中性。ORB/SIFT和平面分支给出了实际低有效率，不强行拒绝身份候选。

## Q9: 身份触发优于固定周期吗？

未证明。同步联合策略和可选独立进程异步worker已经实际接入因果GMT，具有前缀、真实调用和完成时间验证。scene18固定20与联合策略各80次实际VGGT调用，均与因果B0预测完全一致，联合策略单次实测更慢。随机、运动/身份单一触发、多间隔和重复验证矩阵仍在完成。证据：causal_fixed_joint_total80_full_v18_v1.json。

## Q10: 相较 StreamVGGT/FastVGGT 的价值？

不能直接移用长序列加速倍率或流式权重。源码接口和许可证已审计；短双视角速度实测、公平预算对照和真实流式集成仍待完成。

## Q11: 完整 FPS/延迟/显存/参数量？

因果B0 scene18完整实测6.64图像/s；优化异步相机路径5.40图像/s且预测完全相同，0有效几何；固定/联合80次调用为4.95/4.26图像/s。实际硬件为V100，不能称A100结果。静态GMT模型初始化排除，新增模型冷启动及worker启动/关闭算入总成本。三个匹配容量头均约31万参数。详见runtime_results.csv和学习头manifest；不以一次计时宣称稳定加速。

## Q12: 提升是否来自不同检测/配置/评测 bug？

目前没有提升。基线固定checkpoint/config与映射修复，完整五段严格开发集：HOTA=81.0547、CVIDF1=95.6317。OFF、零系数、真实VGGT计算中性对照与整段采样预测一致性通过；B2前两系数首轮输出与B0一致。

## Q13: 三个模块能否分别证明有效？

尚未证明。当前 SCIENTIFIC_SUCCESS、ENGINEERING_COMPLETE、PAPER_READY 均未达成。B0/B1/B2 和几何诊断不能替代完整 B3–B9 消融、因果速度和重复复核。

## Q14: 为什么符合 WWW？

仍有实质适配风险。官方CFP要求明确Web科学问题，仅用数据集或API不充分。实际在线图像读取、GMT目标查询与轨迹片段索引已实现，已完成6个检索评价，其中2个为完整视频。详见retrieval_results.csv；GT只进入事后评价，不进入查询选择/排序。 本地数据集回放不等于互联网来源、真实用户查询或Web部署；见WWW_SCOPE_NOTE.md。

## Q15: 在线视频关联检索的真实贡献？

实际在线图像读取、GMT目标查询与轨迹片段索引已实现，已完成6个检索评价，其中2个为完整视频。详见retrieval_results.csv；GT只进入事后评价，不进入查询选择/排序。 条件Recall@K、完整MRR、检测覆盖、可检索比例、全部请求成功率和本地查询延迟分别报告。完整读帧/检测/GTA/新增特征/索引/查询成本另计；流数量与关键帧预算对照及实质Web系统贡献仍需完成。
