# JEV 开源实现审计

## 目的与边界

本审计服务于 GMT 的长期研究任务：寻找可以迁移到在线多摄像头多目标跟踪决策层的 JEV（typed state-transition decision）实现要素。这里的目标不是把一个固定阈值替换成 MLP，也不是把候选轨迹 ID 当成分类类别；需要的是在运行时接收状态、问题类型和合法动作集合，输出动作分布，并把动作提交后的长期结果纳入训练。

审计只依据核心源码、测试和训练实现，不依据 README 单独下结论。所有仓库均以只读方式浅克隆到 `/tmp/gmt-jev-audit-20261003`，没有加入 WWW 工作树，也没有调用需要 API key 的外部服务。

| 仓库 | 审计 commit | 核心实现位置 |
|---|---:|---|
| `ejhshen/OpenJev` | `e279c44` | `src/openjev/models/decision/head.py`, `src/openjev/models/decision/model.py`, `src/openjev/rlcd/environment.py`, `src/openjev/rlcd/reinforce.py` |
| `jaredpalmer/kev` | `84847f0` | `kev/model.py`, `kev/metrics.py`, `kev/jev.py` |
| `nokia-applied-research/AnyJev` | `e6efe12` | `anyjev/heads.py`, `anyjev/decider.py` |
| `mohit67890/imajev` | `ccf586d` | `vision_decision/contracts.py`, `vision_decision/scoring.py`, `scripts/torch_decision.py` |
| `kshetrajna12/reflex` | `231f896` | `reflex/engine.py`, `reflex/readout.py`, `reflex/calibration_head.py`, `reflex/ensemble.py` |
| `Davipar/djev-dev` | `3ce907e` | `djev/contracts.py`, `djev/engine.py`, `docs/architecture.md` |
| `isHeSatoshi/smalljev` | `65f74b4` | `smalljev/heads.py`, `smalljev/model.py`, `smalljev/confidence.py` |
| `Octalab-Inc/jqv` | `40ac9b8` | `jqv/heads.py`, `jqv/systemone.py`, `jqv/readout.py` |
| `deepanwadhwa/OpenDecision` | `9bbf478` | `src/opendecision/engine.py`, `evidence.py`, `rules.py` |
| `jiangxiluning/Visual-Jev` | `2d68c11` | `src/semif_phase1`, associated readout/reranker code |

## 逐仓库结论

### 1. OpenJev — 最接近可迁移的共享动态动作头

`src/openjev/models/decision/head.py` 的 `DecisionHead` 接收一个问题向量 `[B,H]` 和运行时选项向量 `[B,K,H]`。`question_projection` 与 `option_projection` 是共享投影，`OptionSetInteractor` 在选项维度上做带 mask 的多头自注意力，然后用问题/选项兼容投影和双线性逐选项打分。softmax 只在当前合法选项上进行，K 不需要固定，选项置换不会改变 head 的参数语义。

`src/openjev/models/decision/model.py` 从共享状态/分支的 hidden states 池化问题和每个选项，再通过 `decision_ptr` 送入同一个 head；因此共享 prefix/state 和多问题复用是实现层能力，而不是文档口号。它是静态决策读出，不会生成答案文本。

`src/openjev/rlcd/environment.py` 的 `OutcomeSpec` 支持硬标签或 posterior，`BanditFeedbackEnvironment` 固定 episode outcome，actor 只能看到 handle 和反馈，不直接看到 oracle outcome；`reinforce.py` 使用所选动作反馈、重要性加权和 baseline，`stage2.py` 支持冻结参考分布的 KL。这个仓库是十个仓库中真正实现 outcome-related learning 最完整的一个。

限制：outcome 是单步选项正确性/反馈，不是 GMT 的多帧轨迹效用；没有运行时跟踪状态、反事实 rollout、IDSW/Frag/恢复代价，也没有直接的动作重试语义。

### 2. kev — 条件分支和校准指标较完整

`kev/model.py` 使用 causal packed layout：`[STATE][question branches]`，block-causal mask 使状态共享；`option_isolation=True` 把选项 span 作为独立分支，同一 decision 位置可以比较全部选项。`PointerHead` 用共享 key/query 变换和温度计算

`z_i = (k(h_option_i) · q(h_decide)) / sqrt(r)`，K 可变，候选顺序可以被隔离设计处理。模型支持 state cache/prefix reuse，训练和 serving 的 state/branch 长度约束分开定义。

`kev/metrics.py` 明确实现 NLL、Brier、ECE、confidence/risk-coverage/selective 指标。仓库还包含一个需要外部 API key 的 Jev gateway 评测入口；本审计没有运行它、没有申请或保存 key。

限制：核心代码没有 GMT 所需的长时域状态转移和反事实效用学习；外部 gateway 不是本项目可接受的训练依赖。

### 3. AnyJev — 动态候选与顺序/校准诊断强，但不是共享动作 head

`anyjev/heads.py` 的 `LinearHead` 是按问题拟合的闭式 readout（diff-means/LDA/ridge/RRR），可做 out-of-fold temperature；它要求每个问题单独累积标签，不是一个对所有 question/action 复用的共享 head。

`anyjev/decider.py` 实现 restricted softmax、候选排列边际化、prior correction、共享 state/prefix、多问题和 option permutation；`observe` 在达到 30/60/120 个标签后重新拟合，adaptive shift 会依据 log-margin 停止。这些机制适合校准和顺序不变性实验。

限制：每个问题独立拟合 head，不能直接承担 GMT 中 MATCH/MEMORY/REACTIVATION 三种问题的共同动作语义，也没有长时域 outcome rollout。

### 4. imajev — 类型契约、显式 UNKNOWN 和 abstention 最清楚

`vision_decision/contracts.py` 定义 choice/boolean/ordinal 类型、状态边界和运行时候选；系统显式追加 `UNKNOWN="__unknown__"`。`Result` 区分 `answered` 与 `abstained`，并允许 `insufficient_evidence` 原因；`scoring.py` 校验 logits/probabilities、候选数量和编码，支持候选旋转。

`scripts/torch_decision.py` 的 `enable_readout()` 安装独立的 trainable `Linear[codes, hidden]` readout，可从 LM head 行初始化；训练支持硬 CE、soft `target_probs` 混合、ordinal EMD 和可选 rationale loss。

限制：UNKNOWN 是显式候选而不是 GMT 的 typed state-transition abstention；训练仍是单步标签，没有长期轨迹 utility。

### 5. reflex — 共享状态、候选排列平均和状态条件校准

`reflex/engine.py` 用共享 state prefill/cache 加载多个 question branch；每个 branch 看到状态和自身问题，不看 sibling，支持一次批处理多个问题而不生成文本。`reflex/readout.py` 对候选排列做概率平均，`ensemble.py` 对问题措辞变体集成。

`reflex/calibration_head.py` 预测 log-temperature，输入包含 primitive kind、候选数量、state token 统计、熵和 top margin；可在 held-out hard/soft target 上拟合，并正则到 primitive temperature。它提供了 state-conditioned calibration 和 disagreement 信号。

限制：它不是共享的 GMT 动作 head，也没有 outcome learning 或动作提交后的状态演化。

### 6. djev-dev — 严格 schema 和一次性 typed readout

`djev/contracts.py` 严格限制 choice/noul/score 请求，最多 32 个问题、每题最多 255 个候选；`normalize_logprobs` 和集中度 confidence 只表达读出置信度，源码明确不把它称为 calibration。

`djev/engine.py` 使用固定答案模板和一次 diffusion read，支持合法 label 归一化；架构文档说明 state、image、independent/joint questions 的共享接口。

限制：没有训练用的共享 outcome head、反事实分支或长时域反馈学习。

### 7. smalljev — 适合作为简单 permutation-equivariant baseline

`smalljev/heads.py` 的 `OptionScorerHead` 对每个 option representation 使用共享线性打分，K 可变且对 option permutation equivariant，零初始化给出 uniform；另有固定 max-slot 的 `SlotChoiceHead`、BinaryNoulHead 和 OrdinalScoreHead。`smalljev/model.py` 使用冻结 causal LM、packed state 和 branch isolation，一次 forward 读出 head。

`smalljev/confidence.py` 提供 layer convergence、OOD kNN bank 和 abstention/coverage 曲线。

限制：Slot head 不是动态合法动作语义；整体没有 GMT 长期 outcome training。

### 8. jqv — Pointer head、共享多问题和候选掩码

`jqv/heads.py` 的 `PointerHead` 对动态 K 候选计算

`z_i = (U h_decide) · (V h_i) / sqrt(r) + w · h_i`，再对合法候选 mask；`SlotHead` 则是固定 max choice。`jqv/systemone.py` 在一次调用中共享 state 并决定多个 question，`readout.py` 提供 entropy/confidence/calibration 接口。训练可把 decision head 作为独立 readout/adapter。

限制：没有 long-horizon outcome fine-tuning；需要补上 GMT 的 state transition、legal action 生成和 trajectory utility。

### 9. OpenDecision — 显式证据冲突/unknown，但不是神经共享动作模型

`src/opendecision/engine.py` 是 zero-shot ModernBERT NLI/classifier，可以接受任意运行时 proposition/candidate；`evidence.py` 同时评估 proposition 与显式 contradiction，`rules.py` 有 established/refuted/conflicted/unknown 四值逻辑，`documents.py` 负责检索片段并生成带引用的决定。

限制：confidence 主要是 entropy/concentration，没有 JEV shared neural action head、反事实 outcome 或 tracking-specific state。它的显式 conflict/unknown 契约可借鉴到 disagreement/abstention 诊断。

### 10. Visual-Jev — 视觉输入和条件 option score，但仍是 readout

`src/semif_phase1` 直接从 Qwen native logits 读取运行时选项，支持共享/串行 prefix reuse，并包含 reranker baseline。文档明确说明 conditional option score 不是 calibrated confidence。

限制：没有训练得到的 dynamic action head、长时域 outcome 学习或 GMT tracker state；reranker 也不能直接作为本项目的决策模型。

## 跨仓库能力矩阵

| 能力 | 实际实现最强的仓库 | 对 GMT 的可迁移结论 |
|---|---|---|
| 共享 head + 动态 K + 合法 mask | OpenJev；其次 `jqv`/`smalljev` Pointer/Option head | JEV head 必须接收 runtime legal actions，不能为固定轨迹 ID 设置类别行 |
| question/action 条件化 | OpenJev、kev、jqv | MATCH、MEMORY、REACTIVATION 应是 typed question；动作描述和状态共同决定 logits |
| 共享状态/多问题 | OpenJev、kev、reflex、jqv | 同一 GMT 状态可在一次 forward 中回答多个问题，缓存共享 state |
| 候选置换不变性 | OpenJev、AnyJev、reflex、smalljev、jqv | 训练和评测要做 candidate permutation test，排除位置记忆 |
| 校准 | kev；reflex 的 state-conditioned T；AnyJev 的 OOF T | 必须单独报告 NLL/Brier/ECE/risk-coverage，不把 raw softmax 当 calibrated confidence |
| abstention/unknown | imajev、OpenDecision、smalljev | `ABSTAIN/START_NEW` 要有显式语义和原因，不能隐藏成阈值副作用 |
| outcome learning | OpenJev RLCD/reinforce/stage2 | 可借鉴反馈、importance weighting、reference KL；但 GMT 需要扩展为 H-step counterfactual utility |
| 图像/视觉 readout | djev-dev、Visual-Jev | 视觉输入不是 JEV 成立条件；GMT 的 evidence 由 detector/GMT association 提供 |

## 对 GMT 设计的硬约束

1. **候选 ID 不是动作类别。** GMT 在一个状态下只能生成当前合法动作集合，例如 `ACCEPT_CURRENT`、`REASSOCIATE`、`START_NEW`；JEV 输出动作，而不是输出某个 proposal/track ID。实际 proposal 和轨迹由 GMT 的 association evidence 与后续 assignment 完成。
2. **合法动作集合是运行时输入。** 不同状态可禁止 `REASSOCIATE`、`REACTIVATE_OLD` 或 `WRITE_MEMORY`；softmax/损失都必须使用同一个 legal mask。
3. **共享 head 要对问题类型和 action set 条件化。** 不能为 MATCH、MEMORY、REACTIVATION 各复制一个带固定类别行的 MLP；允许 shared scorer、option encoder、set interaction，但不能把 candidate identity 编进固定输出位置。
4. **校准和 abstention 是单独评估目标。** 原始概率、置信度集中度和 calibrated probability 不能混用；必须保留 held-out calibration、ECE/Brier/NLL 和 risk-coverage。
5. **长期结果必须来自反事实。** OpenJev 的单步 bandit feedback 是可借鉴的训练机制，但 GMT 标签需要从同一状态分支执行 H 帧，使用 AssA/IDF1/IDSW/Frag/contamination/recovery 等长期 utility 构造 soft target。
6. **先冻结 GMT，再训练 JEV。** detector、Stage1 REID 和 canonical Stage2 必须固定；Oracle、阈值、MLP、JEV 共享同一 checkpoint、输入和评估协议。

## 结论

没有一个被审计的仓库可以直接替代 GMT 的 JEV 研究实现。最可复用的组合是：

- 用 OpenJev 的 shared question/option head、合法 option mask 和 set interaction 作为结构参考；
- 用 `kev`/`reflex`/AnyJev 的候选置换、校准、risk-coverage 作为诊断参考；
- 用 imajev/OpenDecision 的 typed unknown/conflict 契约定义显式 abstention；
- 用 OpenJev RLCD 的 feedback/reference-KL 思路，但把 outcome 改成 GMT 的长时域 counterfactual utility。

因此，下一阶段应先完成 GMT OFF/Oracle/固定阈值和强 MLP 对照、建立反事实数据集及指标，再实现最小 JEV action head；当前审计不足以支持直接改写 `run_global_tracker_plus`。
