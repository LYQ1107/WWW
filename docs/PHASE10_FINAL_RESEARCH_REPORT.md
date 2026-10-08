# Phase X 最终科研报告

最终状态：`COMPLETE_SCIENTIFIC_STOP_AT_TINY_GATE`。工程修复、原生H32和v3资格审查完成；固定的三模型共同tiny门槛未通过，后续正式训练与sealed heldout按用户预定规则停止。

221/221原前缀通过，包括原94个失败快照。162事件/840原分支已从真实生产前缀逐个执行，CONTROL/显式KEEP在全部未来窗口的完整状态、事件和RNG与事实流精确匹配。53分支/53事件行的效用或错误持续标签变化（START_NEW52，REASSOCIATE1），旧标签和V–IX结论未覆盖。重新核验训练纠错74、验证59，原组17/9；v3共226行，TRAIN132/validation94。TRAIN Recall@8=97.56%、@16=98.37%、@32/64/Full=100%，未截断候选。

| 目标（共同2000步） | MLP已确认Rank-1 | DeepSets已确认Rank-1 | JEV已确认Rank-1 |
|---|---:|---:|---:|
| conditional_correctness_CE | 50.64% | 86.54% | 86.54% |
| known_edge_binary_CE_diagnostic | 50.64% | 86.54% | 86.54% |
| preregistered_joint_diagnostic | 68.59% | 100.00% | 100.00% |

已完成三种有界目标的真实tiny学习，未完成正式三seed/validation/heldout架构比较。未知选择是未认证，不等于已知GT错误；不能用已知支持上的近零损失证明完整决策可靠。模型输入和归一化没有精确标签冲突，有限值、mask、候选排列、TorchScript及原生分段恢复已检查；保留所有早期故障和checkpoint。本次不会修改未知标签、填零Q、换容易样本、降低95%或扩展Full24来得到通过结果。

## 1_gallery_parity

全部221个原固定前缀通过，包含原94个失败记录；全部162事件/840分支的CONTROL和显式KEEP完整状态、事件、RNG等价。

## 2_H32_changes

53个分支标签变化：52个START_NEW、1个REASSOCIATE。纠错数量重新核验后仍为训练74、验证59；相同数量不代表效用标签未变化。变化来自完整原生状态/转换契约替换旧Replay，不能全归因为某一条Gallery向量。

## 3_MLP

MLP确实完成了真实tiny训练，已知候选与已执行效用对比可以拟合；联合目标下完整合法候选的已确认命中率68.59%，未达95%。不能把未知候选判作错误GT，也不能把它算作正确恢复；真实闭环纠错能力尚未验证。

## 4_DeepSets_vs_MLP

联合tiny下DeepSets达到100%，MLP为68.59%；这仅是同一小样本的拟合差异，没有完整三seed/validation/heldout比较，不能宣称显著优于MLP。

## 5_JEV_vs_ordinary

联合tiny中JEV与DeepSets均为100%，没有已观察到的额外tiny优势。正式相同预算的闭环比较尚未执行，JEV独特结构优势未得到证明。

## 6_supervision

正确性CE和已知边二元CE都不能使三模型达到完整合法候选门槛；原计划的联合监督使DeepSets/JEV通过该tiny指标，但MLP仍失败。可以记录其小样本作用，不能据此选择最优泛化监督。

## 7_Unified_JEV

目前没有足够证据扩展MATCH–MEMORY–REACTIVATION Unified JEV。下一步应先审计未锚定候选的身份语义与可验证监督支持；不得用GT泄漏、把未知改负、重新挑选有利tiny组或降低门槛来继续。

复现入口：`capture_jev_phase10_production.py` → `check_jev_phase10_prefix_parity.py` → `run_jev_phase10_native_forks.py` → `finalize_jev_phase10_data.py` → `tiny_jev_phase10_candidates.py`；两次有界目标诊断由`diagnose_jev_phase10_tiny_objective.py`及其冻结protocol绑定。精确运行源码根、commit、GPU、seed、config/cache/foundation/B2 SHA与checkpoint路径均在JSON manifest。large native states、数据张量、完整分支日志和tiny权重留在独立服务器runtime；GitHub仅提交必要代码、compact指标、文档和SHA。
