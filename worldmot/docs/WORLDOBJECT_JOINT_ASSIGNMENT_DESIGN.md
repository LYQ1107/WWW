# 历史对象中心的两摄像头联合分配

实现：`worldmot/tracking/joint_world_object_assignment.py`。

历史对象 q 的选项为 NULL、(A_i,NULL)、(NULL,B_j)、(A_i,B_j)。新对象允许高置信双视角联合
Birth；剩余观测是 singleton Birth/未确认对象。所有选项在同一冲突图中竞争，没有预先不可逆配对。

每个选项携带历史 Global ID（Birth 为 None）、当前观测索引、benefit 和原因。
优化最大化所选 benefit 之和，约束每个历史 ID 最多选一个选项，每个检测最多出现一次，
同一对象每摄像头最多一个观测。NULL 的相对 benefit 为0；未选观测仍输出真实框。

## 候选与证据

历史—观测证据包括真实身份 embedding 余弦、联合协方差下的空间距离、几何有效性、
漏观测历史及当前可靠性。学习头接收相机来源、可见性和时间间隔等上下文。
位置不符但对象已经失踪、当前 ROI 与身份极强一致时允许 `VISUAL_RECOVERY_SPATIAL_CONFLICT`；
此时不会将冲突位置或特征直接覆盖旧状态。

学习后的单视角收益为：

```text
(1-w) * heuristic_evidence
  + w * tanh(assignment_logit[q,d] - null_logit[q])
  + 0.1 * tanh(existence_logit[q])
```

w 默认0.75，限制在[0,1]。因此强 NULL 可以否决启发式匹配，而不是仅附加一个无法改变决策的 gate。
双视角选项共同检查当前 A/B 的视觉、空间兼容，并附加有限的一致性收益。
Full 方法的双视角 Birth 要求可靠几何与高 ROI 一致性；无几何方法使用更严格视觉阈值。
几何中心近不构成无条件合并理由。

每个历史对象每摄像头仅保留 Top-K=4 个候选，所以最多生成 `2K+K²` 个匹配选项，另有 NULL。
新生配对每个 A 也限制 Top-K。候选前的比较量有显式观测/历史上限，不无边界枚举历史×A×B。

## 求解与复杂度

共享检测的选项组构成连通分量。

- 选项组合数（含 NULL）≤4096：精确枚举，不截断可行状态；只对剪枝后的有限假设集保证最优。
- 更大分量：Beam=64，每层保留确定性排序后的最优可行状态。没有全局最优保证。
- 相同分数按 Global ID/规范化 detection key 打破平局，输入字典顺序不影响结果。
- 历史/观测超过1024：显式审计的 singleton 未确认新生回退，保留框。
- learned dense pair 数超过65,536：推理使用已记录的规则证据回退；训练报错，不删除监督样本。

设选项组数 G，每组最多 O 个选项，Beam 宽度 W，则大型分量扩展量有 `O(G*W*O)` 上限。
诊断输出 component 模式、宽度、扩展状态数、候选比较数、回退原因和约束验证结果。
实现仅要求 NumPy，没有 ILP 服务器或隐藏优化器，也没有调用 association.py 的 greedy 降级。

## 分配后的更新

`assigned_measurement()` 在身份决定之后可靠性加权融合空间与 ROI。
几何失败时 measurement 不含可靠空间；Bank 可更新真实视觉证据并保留旧世界状态。
Update Gate 只控制已匹配对象的状态写入，不代替身份分配；Memory Delta 写入该 Global ID，后续 forward
直接读取更新后的 embedding。Birth 概率控制新生确认程度和联合 Birth 候选，Existence 控制失踪对象保留。

为 E0/E1 保留的 `legacy_worldobject_builder_v1.py` 和 `legacy_worldobject_bank_v1.py` 是从
`190c735` 固定下来的显式对照。FEATURE_OFF 的零视觉行为只允许出现在这个命名对照中；
它不代表 V3 身份通路正常。E1 的旧/新 Bank 记忆更新也不同，因此其对比是架构整体对比，
不能单独归因于求解器。旧实验数值保留原标签，不因这里提供对照入口而变成新结果。
