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

### 评估接口发现及保留式复核

上方原始表格保留 inherited assess 数字。该函数仍解析旧模块 batch，没有早期原型/affinity 候选。原训练、certified probe、全量同状态诊断与实际在线采用 recovery batch，未因此漏用证据。修正是隔离克隆 evaluator 的全局 batch，重评同一个 checkpoint；没有改权重、标签或 Gate。新旧方案集合改变，所以新旧排名也不是相同方案集合的因果对照。

| 完整 R2 输入 | Cross 正例正确/总数 | Persistent 正例正确/总数 | 排名 |
|---|---:|---:|---:|
| jev | 940/1001 | 2889/2994 | 0.840241 |
| threshold | 879/1001 | 2459/2994 | 0.880934 |
| mlp | 938/1001 | 2856/2994 | 0.876413 |
| attention | 920/1001 | 2662/2994 | 0.874906 |

原945对702、2895对1261不能直接代表实际完整R2输入下的差距。完整输入仍须检验辅助预测能否改变已认证正确身份。接口修正的首个Tiny失败发生在梯度前，日志保留；修正后的融合器用v2新ID/目录。等待式finisher曾因失败守卫暂停，审核后原脚本恢复，没有中断原神经实验。

| 干预 | N01 | N10 | 净纠错 | 对完整模型改配行数 | WHO-only 排名 |
|---|---:|---:|---:|---:|---:|
| cross_representation_only | 51 | 52 | -1 | 569 | 0.838655 |
| full | 238 | 131 | 107 | 0 | 0.836816 |
| mean_representations | 11 | 14 | -3 | 697 | 0.834222 |
| no_joint_plan_residual | 238 | 131 | 107 | 0 | 0.836863 |
| persistent_representation_only | 41 | 9 | 32 | 584 | 0.842145 |
| zero_auxiliary_logits | 238 | 131 | 107 | 0 | 0.836816 |

完整模型中两辅助问题均正确、最终身份错误且存在合法正确替代的行数：188。没有合法正确替代的对应行数：475；后者不定义为排名失败。

已认证正确合法身份未进入最终方案的行数：1883。UNKNOWN 比较：6105。完整模型正负辅助交叉表与最多 40 个事件的 ID、历史可用性、早期外观余弦、原始/修正分数保存在诊断 JSON。

把两个辅助输出置零后最终分数逐位相等：最终评分没有直接读取辅助 Head 输出。这个结果证明直接输出路径缺失，不能证明辅助训练没有通过隐藏表示起作用。屏蔽／平均的差异检验 B；两问题正确但合法纠错仍失败检验 A；覆盖、UNKNOWN、认证与依赖审计检验 C。三种原因可能共同存在，不能仅由一个比例排他归因。

此 checkpoint 删除联合方案残差没有改变选中方案；在该全量状态样本上不能把当前身份收益归因于联合方案残差。完整融合比单路或平均的冻结干预更好，只证明本权重中组合表示的作用，不能替代同预算重训的架构对照。相同全量源状态下普通 MLP 净纠错151、阈值119、普通 Attention147，JEV107；没有建立 JEV 独立优势。

## 第5/6项：重新训练的最小融合对照

同一冻结 R2 JEV 前端和辅助 Head，四个最终融合器：显式辅助概率门控、两个独立最终 Reader、普通 MLP、普通 Attention。新增参数81786/82274/82476/81826，差距小于1%。同样两路表示、两预测概率、原余量和因果早期外观，同批次与损失；Tiny128 后 fresh Pilot1000，再做全部 TRAIN16 原生在线和身份提交审计。

这只对比相同已训练 JEV 前端上的最终融合，不代表从头训练的普通控制器架构。原纯 MLP/Attention 主对照继续独立报告。新增融合梯度只有1000步，属于已允许 R2 范围；既有正式4000和原晋级条件均不变，没有第四个大型版本，也没有 DEV 调参。

| 融合器 | Tiny | Pilot | 同状态净纠错 | 完整在线 |
|---|---|---|---:|---|
| explicit_scores | COMPLETE | PENDING | PENDING | PENDING |
| independent_readers | COMPLETE | PENDING | PENDING | PENDING |
| mlp | COMPLETE | PENDING | PENDING | PENDING |
| attention | PENDING | PENDING | PENDING | PENDING |

## 冻结干预的实际原生状态验证

H64使用事前固定的真实XV native prefix32/0，五方法起始state/input/candidate校验，全部未来动作与Gallery/Bank状态重新执行；这是配对干预，不是独立全视频。完整TRAIN16冷启动干预另跑，开始状态相同，之后状态自然分叉，不能再声称全程输入相同。

| 冻结干预 | H64已知错误观测 | 完整TRAIN16 HOTA | AssA | IDSW |
|---|---:|---:|---:|---:|
| full | PENDING | PENDING | PENDING | PENDING |
| cross_representation_only | PENDING | PENDING | PENDING | PENDING |
| persistent_representation_only | PENDING | PENDING | PENDING | PENDING |
| mean_representations | PENDING | PENDING | PENDING | PENDING |
| no_joint_plan_residual | PENDING | PENDING | PENDING | PENDING |

原生提交、初始状态SHA及每步预测源文件在question_intervention_paired / question_intervention_native报告。H64统计遵循冻结first-known-owner口径，与同状态anchor N01/N10证书不同，不能混合成一个纠错指标。40个源状态错误例并非已经在H64窗口逐个提交；完整状态机制验证与例案表须分别解读。

## 原 R2/R3 晋级与强基座

继续全部预注册 R2 消融、完整 TRAIN16 和既有机制 Gate，再执行强 XIII Fixed 原始及阈值／MLP／Attention／JEV 增量。同状态 N01/N10 与完整 mutated-state 指标分开报告。两 Gate 失败也不反复调 DEV17/18/19，不修改阈值求成功。强基座历史参考不是本次已完成同基座结果。

Stage2、真实 FPS、参数与三种子正式指标以原 finisher 完成文件为准。此报告不会把 pending 值填成0，也不会宣称独立验证或研究突破。
