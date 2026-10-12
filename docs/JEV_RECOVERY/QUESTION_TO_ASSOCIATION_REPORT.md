# Question 到实际身份关联：补充诊断

科学状态：`DEVELOPMENT_NOT_ESTABLISHED`。辅助分类提高不计作主目标成功。原 Gate、原训练标签与权重保留。

## 当前标签与科学口径

实际 load_records 先读取旧 R2 Cross/历史可靠性标签，再用 first-past-certified known_options/positive 替换 Stability。当前 Persistent 是与第一次过去认证的 ID 归属是否一致。两版分歧统计不能直接比较。混合历史可导致旧可靠性=0；当前已经绑定的过去证书保留。未绑定 ID、未知当前人保持 UNKNOWN。证书不代表真实出生 owner 或未来安全。

训练的 3601 个局部冲突组按共享人员／ID 历史连接后只有 3 个保守依赖分量。不能把相邻帧和候选边当作独立纠错事件。标签源 SHA 与实际 loader 见 QUESTION_LABEL_SEMANTICS_AUDIT.json。

## 复核用户给出的 Pilot 现象

| 模型 | Cross 正例正确/总数 | Persistent 正例正确/总数 | 当前排名准确率 |
|---|---:|---:|---:|
| jev | 945/1001 | 2895/2994 | 0.835979 |
| threshold | 702/1001 | 1261/2994 | 0.853175 |
| mlp | 930/1001 | 2564/2994 | 0.841270 |
| attention | 769/1001 | 1951/2994 | 0.846561 |

这里的当前排名 targets 包含 0.25 Persistent 偏好；新诊断另报纯 WHO 排名。这些是固定 128 个复用 TRAIN16 状态的相关样本，不是完整 MOT 或独立验证。

## A/B/C 的冻结权重诊断

同一个 JEV Pilot checkpoint，全部 2067 个真实 TRAIN16 状态与相同合法方案；不写原生状态。屏蔽的是最后融合的隐藏表示，包括 private DEFER 表示，类型向量和权重保留。简单平均使用原训练评分头，属于分布改变的冻结干预，不代表重新训练的平均架构。

冻结干预正在执行；尚未生成完成结果。

## 第5/6项：重新训练的最小融合对照

同一冻结 R2 JEV 前端和辅助 Head，四个最终融合器：显式辅助概率门控、两个独立最终 Reader、普通 MLP、普通 Attention。新增参数81786/82274/82476/81826，差距小于1%。同样两路表示、两预测概率、原余量和因果早期外观，同批次与损失；Tiny128 后 fresh Pilot1000，再做全部 TRAIN16 原生在线和身份提交审计。

这只对比相同已训练 JEV 前端上的最终融合，不代表从头训练的普通控制器架构。原纯 MLP/Attention 主对照继续独立报告。新增融合梯度只有1000步，属于已允许 R2 范围；既有正式4000和原晋级条件均不变，没有第四个大型版本，也没有 DEV 调参。

| 融合器 | Tiny | Pilot | 同状态净纠错 | 完整在线 |
|---|---|---|---:|---|
| explicit_scores | COMPLETE | PENDING | PENDING | PENDING |
| independent_readers | PENDING | PENDING | PENDING | PENDING |
| mlp | PENDING | PENDING | PENDING | PENDING |
| attention | PENDING | PENDING | PENDING | PENDING |

## 原 R2/R3 晋级与强基座

继续全部预注册 R2 消融、完整 TRAIN16 和既有机制 Gate，再执行强 XIII Fixed 原始及阈值／MLP／Attention／JEV 增量。同状态 N01/N10 与完整 mutated-state 指标分开报告。两 Gate 失败也不反复调 DEV17/18/19，不修改阈值求成功。强基座历史参考不是本次已完成同基座结果。

Stage2、真实 FPS、参数与三种子正式指标以原 finisher 完成文件为准。此报告不会把 pending 值填成0，也不会宣称独立验证或研究突破。
