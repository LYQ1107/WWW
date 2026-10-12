# Codex 长周期执行任务：让 JEVTrack 真正取得性能突破

## 0. 角色、仓库与最终目标

你是本项目的首席科研工程师，负责算法设计、源码修改、数据分析、模型训练、GPU实验调度、真实性审核和GitHub持续交付。

项目：JEVTrack

研究任务：Online Multi-Camera Multi-Object Tracking

GitHub：

https://github.com/LYQ1107/WWW

当前研究分支：

jev/www-jev-oneweek-20261012

当前已核查 HEAD：

dbfc79d20a5ac003d6f3fd23137c16f465acb8d0

原工作目录：

/home/liuyeqiang/WWW_jev_oneweek_20261012

原实验目录：

/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1

新建后续研究分支：

jev/www-jev-recovery-20261012

建议新工作树：

/home/liuyeqiang/WWW_jev_recovery_20261012

建议新实验目录：

/data1/liuyeqiang/WWW_jev_recovery_runtime/20261012_v1

执行之前先检查GitHub最新SHA与本地状态。若已有更新，阅读新增报告，保留已经完成的实验，不覆盖历史结果，也不要让正在运行的原始效率任务失效。

### FINAL RESEARCH GOAL

**让 JEVTrack 在真实在线多摄像头多目标跟踪中，取得相对于最强普通关联方法的稳定、可复现增益。**

主要研究目标：

1. 提高跨摄像头 Global ID 关联质量。
2. 提高长期 Global ID 稳定性。
3. 降低错误身份合并、身份跳变及历史污染。
4. 使多问题 JEV 确实提供普通阈值、MLP和标准Attention难以获得的增量信息。
5. 控制新增参数量和FPS损失。
6. 形成能够接受WWW审稿人检验的实验与方法证据。

这里的最终Goal不是“完成诊断”，也不是“训练3个模型”。

你应该主动发现目前影响性能的原因，修改模型，执行受控实验，根据实际证据继续优化。

**但是不得为了让Goal变成成功而伪造结果、修改旧实验、挑选有利种子、泄漏TEST或事后放宽科学评价标准。**

如果一种方案失败，停止升级这种方案，保存负结果，进入下一条有依据的研究路线。

如果有限预算内所有路线均失败，仍完成充分的机制解释和下一步研究建议，不得虚构正向成果。

---

## 1. 先完整继承当前研究结果

不得重新从零实现GMT。

优先复用：

gtr/modeling/jev_oneweek/model.py

gtr/modeling/jev_oneweek/evidence.py

gtr/modeling/jev_oneweek/plans.py

gtr/modeling/jev_oneweek/native.py

reproduction_tools/jev_oneweek_learning.py

reproduction_tools/jev_oneweek_supervision.py

reproduction_tools/jev_oneweek_train.py

reproduction_tools/jev_oneweek_online.py

reproduction_tools/jev_oneweek_paired_native.py

reproduction_tools/jev_oneweek_identity_analysis.py

reproduction_tools/jev_oneweek_efficiency_queue.py

同时复用：

Phase XIII Fixed Question正式checkpoint。

Phase XIV Multi-question正式checkpoint。

Phase XV V3冻结checkpoint。

原始GMT完整Stage2权重及已验证的原生执行路径。

### 当前冻结事实

三种子完整DEV17/18/19在线结果：

Phase XIV parent：
HOTA 73.505；
AssA 73.639；
CVIDF1 88.546；
IDSW 391.7。

Pair learned threshold：
HOTA 73.752；
AssA 74.134；
CVIDF1 87.795；
IDSW 115.7。

普通MLP：
HOTA 73.493；
AssA 73.582；
CVIDF1 87.286；
IDSW 114.0。

当前JEV：
HOTA 73.374；
AssA 73.360；
CVIDF1 88.092；
IDSW 123.3。

仅Stability模块：
HOTA 73.567；
AssA 73.748；
CVIDF1 88.088；
IDSW 125.0。

历史Phase XIII Fixed：
HOTA 75.637；
AssA 77.907；
CVIDF1 91.889；
IDSW 105。

原始完整GMT：
HOTA 78.110；
使用不同完整感知及Stage2前端，不得直接当作同输入模块增益比较。

当前JEV新增169,092参数。

当前JEV真实场景FPS约3.703。

同前端普通MLP约3.773 FPS。

JEV相较该MLP的FPS低约1.9%。

当前JEV独立结构增益状态：

NOT_ESTABLISHED。

### 特别提醒

Phase XIII Fixed历史上优于当前JEV。

不能只追求超过73.752而忽略已经达到75.637的历史强方法。

但是，也不能把不同初始化、不同训练阶段的历史数值当成同条件公平比较。

我们的实验分成两个层次：

**一级：同冻结主干的增量架构公平比较。**

**二级：当前最强完整跟踪系统上的实际性能突破。**

两级结果分别报告，不得混淆。

---

## 2. 建立新的持续科研控制器

新增：

docs/JEV_RECOVERY/MASTER_GOAL.md

docs/JEV_RECOVERY/HYPOTHESIS_REGISTRY.md

docs/JEV_RECOVERY/EXPERIMENT_ROADMAP.md

reports/JEV_RECOVERY/MASTER_GOAL.json

reports/JEV_RECOVERY/EXPERIMENT_REGISTRY.json

reports/JEV_RECOVERY/NEXT_ACTION.json

reproduction_tools/jev_recovery_controller.py

永久Master Goal：

Reliable Cross-Camera Identity Association with Evidence-Grounded JEV

具体含义：

使JEV根据跨摄像头身份一致性和历史身份可靠性作出更准确的关联决策，并在真实在线状态更新下保持长期身份稳定。

所有实验围绕两个主要创新方向：

A. Cross-Camera Identity Consistency。

B. Persistent Identity Consistency。

不要继续为了形式增加第三个创新点。

允许内部使用Question、Option、Plan、Owner等术语，但论文贡献表述必须保持简洁、与多摄像头跟踪直接相关。

研究控制器必须将每个方案分为：

ENGINEERING_PASS/FAIL

SUPERVISION_ELIGIBLE/INELIGIBLE

ASSOCIATION_IMPROVED/NOT_IMPROVED

CROSS_CAMERA_IMPROVED/NOT_IMPROVED

JEV_INDEPENDENT_VALUE_ESTABLISHED/NOT_ESTABLISHED

EFFICIENCY_PASS/FAIL

MASTER状态不能因某一阶段结束就自动COMPLETE。

---

## 3. 第一个核心问题：为什么两个Question没有发挥作用？

这是本轮最高优先级。

现有事实：

801,940个双问题标签都已知的候选边中，只有270个两个问题的目标不同。

即约0.034%。

已暴露的TRAIN16中，两问题共同可认证且目标不同的边数量为0。

在106个已有TRAIN保留时间块问题分歧样本上：

JEV两个问题同时答对79个。

普通Attention答对99个。

MLP答对105个。

Pair阈值答对105个。

这说明：

当前数据虽然允许两个Question分别训练，但它们多数时候接受几乎一样的监督。

### 必须执行的分析

对全部TRAIN事件建立Question Disagreement Census。

区分：

1. Cross与Stability都正确。
2. Cross正确、Stability错误。
3. Cross错误、Stability正确。
4. Cross与Stability均错误。
5. 其中一个问题UNKNOWN。
6. 两个问题均UNKNOWN。
7. 当前关联存在合法竞争身份。
8. 真实正确身份没有出现在当前合法池中。

按以下层级去重：

video；
GT person；
camera pair；
Global ID conflict；
64-frame temporal block；
历史错误事件依赖组。

不要把几十万相邻候选边当成几十万个独立身份冲突样本。

建立：

QUESTION_DISAGREEMENT_CENSUS.json

QUESTION_EFFECTIVE_SUPERVISION.json

QUESTION_LABEL_CONFLICTS.md

分析两个Question是否真的需要不同的证据。

如果Cross与Stability在真实样本里几乎恒等，应重新定义具有不同监督语义的问题，而不是简单给同一个任务换两个名字。

可以探索：

Question A：
当前目标是否具有可靠的跨摄像头同人证据？

Question B：
当前具体Global ID是否具有可靠的连续归属证据？

二者不能被同一个外观相似度直接决定。

不能为制造Question差异而随意修改GT标签。

如果真正存在的独立差异不足，记录监督上限，并进入后续关联结构改进；不必无限等待这两个Head被训练好。

---

## 4. 第二个核心问题：修复Question信息过早融合

检查当前源码：

gtr/modeling/jev_oneweek/model.py

当前逻辑包含：

cross, stability = features

feature = (cross + stability) / 2

两个Question的表示在最终关联前被直接平均。

新的实验不能只保留这种融合。

### 设计至少三个同参数预算对照

F0：
现有简单平均。

F1：
跨摄像头证据与身份稳定性分开保留，使用普通MLP进行融合。

F2：
真正Question-conditioned JEV读取。

F2要求：

Shared State编码真实共享。

Cross Question读取跨摄像头历史。

Stability Question读取身份连续性历史。

同一个具体Global ID的两种证据保留为不同类型。

最终Reader能感知两种证据何时一致、何时冲突。

最终输出依赖具体问题与合法身份动作，而不是仅依赖一个平均向量。

需要设计能够单独验证的问题输出：

Cross-Camera Consistency Score。

Identity Continuity Score。

Association Decision Score。

没有可靠监督的概率不能自动视为已校准的正确性概率。

### 硬性消融

固定Question替换动态Question。

移除Cross Question。

移除Stability Question。

两个Reader分别训练但不交互。

普通MLP融合两个Question。

标准Attention融合两个Question。

真正条件化JEV融合两个Question。

如果F2效果不如F1，不得宣称JEV多问题结构具有独立优势。

---

## 5. 第三个核心问题：JEV是否有机会看到正确的关联？

检查：

gtr/modeling/jev_oneweek/plans.py

当前逻辑：

使用冻结基础匹配分数生成最多4个完整关联方案。

主要从原始已有匹配和近分数竞争关系生成替代。

部分原始DEFER行不会主动形成新的现有ID关联尝试。

这可能导致：

正确身份确实位于合法active池内，但不在实际供JEV评价的少量方案中。

此时，无论Reader训练多好，都无法产生真正纠正身份的方案。

### 先做GT-free Candidate Coverage Audit

必须区分：

A. 正确身份不在原始active MATCH池。

B. 正确身份在active池，但不在当前生成的4个合法方案。

C. 正确身份已经在某个合法方案中，但JEV评分选择错误。

D. 正确身份只有在Bank恢复阶段才可使用。

E. 历史owner无法认证，不能判定当前选择是否正确。

F. 存在多个同人ID别名，不能把所有变化自动算纠错。

GT仅用于离线事后审计。

绝不能在实际推理时利用GT保证正确身份进入方案。

### 设计新的合法方案生成策略

P0：
原始4方案生成。

P1：
显式处理DEFER与合法既有ID的竞争关系。

P2：
在多个检测争用同一Global ID时，生成合法的竞争分配方案。

P3：
在相近身份分数与历史冲突情况下，形成有界的替代关联。

可以比较最多4和8个合法方案的覆盖率与推理开销。

全部方案必须符合：

当前真实合法身份池；
每摄像头身份唯一约束；
真实存在的跨摄像头共享Global ID；
原生Bank恢复语义；
真实DEFER行为；
没有GT或未来帧；
没有虚构旧身份。

需要单独比较：

正确关联的理论可达率；

方案集合实际覆盖率；

JEV在覆盖后的正确选择率；

额外计算成本。

如果P1/P2改善覆盖，必须让MLP、阈值与Attention获得同样的扩展方案集合。

不能把候选生成提升伪装成JEV结构优势。

---

## 6. 第四个核心问题：模型没有学到真正的联合决策能力

当前JEV具有非加性联合评分计算，但有效监督偏少。

训练日志显示：

三个种子正式4000步中，带有效联合排名监督的batch约为453、510、479。

多数batch不具备可靠的完整分配排名标签。

现有TRAIN16方案完整认证覆盖不足。

### 不要继续盲目加训练步数

首先改进监督的有效性。

要求：

在TRAIN中利用真实当前检测GT与真实过去身份归属建立监督。

对一个完整分配内仅部分可认证的情况：

保留UNKNOWN；

不要求所有未改变的行重新认证；

只有发生变化的行都可认证时，才允许形成对应局部比较；

对于涉及其他检测的容量冲突，必须检查所有受到影响的改变行。

可以考虑局部关联冲突组的结构化学习：

例如两个检测与两个历史ID发生真实竞争。

仅对具有明确身份监督的局部冲突进行联合评分学习。

不得用未执行的未来动作效用代替当前身份正确性标签。

不得将一个局部GT正确方案称为长期安全动作。

### 新的训练评价

必须报告：

可认证训练冲突组数量；

有效独立人物/视频组数量；

每个epoch中联合排名损失非零的batch比例；

真正提供梯度的独立冲突组；

当前正确方案覆盖率；

当前关联N01/N10；

非加性Reader是否真的影响最终选择；

是否只在UNKNOWN事件产生大量无从认证的改配。

如果训练信号仍几乎全部为普通边匹配，就优先学习普通关联，不要额外声称联合决策已成立。

---

## 7. 第五个核心问题：为什么不用更强的Phase XIII Fixed作为起点？

这次必须实施双基座实验。

### Track A：Phase XIV Multi-question

作为当前结构的公平延续。

优点：
已有动态问题结构；
已有三种子；
已有完整同条件强基线。

缺点：
HOTA73.505；
历史IDSW较高；
当前新JEV继续改善空间有限。

### Track B：Phase XIII Fixed Question

使用已有正式20k三种子checkpoint。

历史结果：

HOTA75.637。

AssA77.907。

CVIDF1 91.889。

IDSW105。

它应当成为这次性能突破路线的重要基座。

在不破坏原关联器的条件下，尝试接入新JEV的跨摄像头证据与身份稳定性Reader。

特别注意：

不同基座不能直接比较新增Head结果并归因于JEV。

必须在Track B上同时训练：

Fixed + Pair Threshold。

Fixed + MLP。

Fixed + Attention。

Fixed + JEV。

所有模型使用相同的冻结Base、训练数据、合法候选、监督标签与训练预算。

新模块初始化应当严格保持原始Base输出。

不要随机添加一个Reader后直接全量替换Fixed的关联分数。

### Track C：原始完整GMT/GTA，条件满足后再执行

原始GMT开发集HOTA78.110。

这是不同完整前端，具有真实的1152D特征与GTA模块。

如果Track A/B仍无法获得足够好的系统性能，但存在可靠JEV局部证据，可以研究在完整原始GMT后增加小型JEV身份修正。

前提：

实际恢复原GMT输入、权重及关联路径；

确保所有强对照也拥有相同输入和修正接口；

不能将额外ROI/RPCE证据当作JEV架构收益；

同时报告原始GMT和GMT+JEV的完整FPS与参数净变化。

只有Track A/B提供积极机制证据，且资源允许时才启动Track C。

---

## 8. 挖掘JEV可能真正具有的跨摄像头优势

本轮存在一个不能忽视的正向信号。

当前DEV历史身份检索：

JEV Precision约98.04%。

Pair阈值约95.18%。

JEV Recall约87.68%。

Pair阈值约86.08%。

但JEV的整体HOTA和IDSW不如Pair阈值。

必须分析这两组指标为什么不一致。

建立：

CROSS_CAMERA_ADVANTAGE_AUDIT.json

分析：

哪些摄像头之间的身份检索得到改善？

哪些具体Global ID产生更少错误历史返回？

JEV是否减少了错误跨相机关联，但增加了单摄像头身份碎片？

更高Precision是否伴随不同的查询覆盖、历史长度、空返回率？

JEV是否在较少数长身份轨迹上改善明显，而在其他人物上退化？

是否由同一人物的长时间重复查询放大了平均收益？

报告按：

视频；
人物；
身份冲突组；
摄像头组合；
实际查询事件。

分别统计。

必须比较：

JEV；
Pair Threshold；
MLP；
Attention；
Phase XIII Fixed；
Phase XIV Parent。

特别建立固定查询分母的受控检索比较。

不能让不同策略仅因生成不同数量的查询就制造更高Precision。

不能用数百万相关返回直接声称数百万独立纠错。

如果JEV确实减少了跨摄像头身份污染，应考虑将其作为主要研究目标之一，但HOTA、AssA、CVIDF1、IDSW和检索覆盖必须同时报告。

---

## 9. 三条有界算法改进路线

本次允许至多三个主要科学版本，每个版本先用廉价实验判断是否值得升级。

### Revision 1：修复关联可达性与有效监督

目标：

改善合法关联方案覆盖；
改善局部冲突监督；
减少UNKNOWN中无意义干预。

不改大模型结构。

先冻结方案生成和标签协议。

对同样的扩展候选，同时比较：

Pair Threshold；
MLP；
Attention；
现有JEV。

如果改进来自候选覆盖，准确归因给候选生成。

这一版本的价值是让后续JEV真正拥有可学习的难例，而不是继续在缺乏有效监督的方案上训练。

### Revision 2：真正的问题条件化跨摄像头JEV

目标：

解决两个Question输出被直接平均的问题。

引入分别表达跨摄像头支持与历史身份连续性的Reader。

重点处理两类信息冲突：

跨摄像头相似但历史身份不稳定；

当前摄像头身份连续但跨摄像头证据矛盾。

在已认证冲突事件上学习，而不是大量重复普通容易关联。

用动态Question、固定Question、普通MLP、标准Attention进行同预算比较。

必须报告：

两个问题的目标分歧；
各自正确率；
共同正确率；
联合关联改进；
QUESTION消融；
新增计算成本。

### Revision 3：强基座身份稳定JEV

目标：

将前两个版本中真正有效的部分迁移到Phase XIII Fixed强基座。

只保留实际有增益的模块。

不得机械拼接Revision 1、2全部设计。

先建立强基座原始输出完全等价性。

再比较：

强Fixed原始；
强Fixed + Pair Threshold；
强Fixed + MLP；
强Fixed + JEV。

如果在强基座上没有增益，需要明确判断：

是方法冗余；
是监督失效；
是可合法纠错机会已很少；
还是新结构破坏了原有强关联分数。

优先通过对照定位，而不是扩大网络。

---

## 10. 训练计划与自动执行

原研究只有一周时间。

请在当前剩余周期内自主安排GPU与任务队列。

允许本次新研究在原已经获准的MATCH-only TRAIN12/13/14范围内进行有界后续训练，TRAIN16仅作既有暴露审计。

不自动扩张其他数据集授权。

所有试验必须先创建新的预注册配置与实验ID，保持历史旧Gate不变。

### 每个Revision执行以下顺序

Step 1：
修复模型/数据接口，运行单元及原生parity。

Step 2：
Tiny128，只检验可学习性。

Step 3：
Pilot1000，检查有效困难样本及有无严重身份错误。

Step 4：
有机制证据后，开展有限正式训练。

建议正式训练最多累计4000–8000 updates/seed。

不能把旧4000-step结果与新8000-step结果直接叫作同预算公平对照；新增预算也要提供给主要普通控制。

Step 5：
在原已暴露TRAIN控制器审计范围内判断机制是否可行，冻结实际模型与超参数。

Step 6：
对冻结合格模型运行完整DEV17/18/19原生在线评价。

Step 7：
若有前景，再补三个固定种子与所有强控制。

Step 8：
判断是继续下一Revision、保留当前最佳模型，还是科学路线需要转向。

### 有效资源使用规则

优先复用：

现有Stage1；
真实ROI/ReID特征；
Phase XIII/XIV/XV checkpoint；
真实native轨迹；
缓存；
既有官方MATLAB与TrackEval；
历史训练脚本。

不要重复下载数据集。

先检查GPU空闲与其他用户进程。

不得停止其他任务、强行抢占GPU或修改代理和系统服务。

已有原一周效率测试继续保留，不能因为新实验而中断或覆盖其输出。

新实验使用独立命名空间和checkpoint。

禁止按DEV结果挑选最优seed或多次调参。

正式比较固定使用原三个种子：

20261008；
20261009；
20261010。

不允许只报告最有利的一个种子。

---

## 11. GOAL达成标准

设置三个逐级目标。

### Goal A：超越目前最强普通增量方法

当前同基座强对照为Pair Threshold：

HOTA73.752；
AssA74.134；
IDSW115.7；
CVIDF1 87.795。

新JEV第一阶段探索目标：

HOTA至少提高约0.3–0.5个百分点；

AssA同步改善；

CVIDF1不降低；

IDSW不出现明显退化；

有可认证真实纠错增量；

性能不是由极端一个种子主导。

以上为新研究设计目标，不是事后可调整的成功声明。

必须另验证普通MLP和Attention不能以相同证据和预算获得相同改善。

### Goal B：达到历史强Fixed基座水平，并证明新增价值

历史Phase XIII Fixed HOTA75.637。

希望新JEV至少能够保留强Fixed原有性能，并争取在相同基座和公平控制下取得额外提升。

如果仅仅超过73.752，却远不如75.637，不能将整体系统称为明显优于现有最强方案。

必须披露历史阶段差异。

### Goal C：WWW级独立贡献

同时具备：

多摄像头Global ID关联增益；

长期身份稳定性证据；

跨摄像头索引质量改善或明确稳定性机制；

五个审稿人问题的公平对照；

新增参数和真实FPS完整报告；

JEV多问题结构确实发挥独立作用；

独立验证所需数据资格满足。

DEV17/18/19已在旧研究中反复使用，不能称为独立测试。

20/21/22、official TEST和Full24继续封存，除非之后获得与原协议一致的新授权。

如果没有独立合格验证，最多标记为DEVELOPMENT_GO，不得声称已经证明泛化或WWW录用级结果。

---

## 12. 必须继续回答五个审稿人问题

新建：

docs/JEV_RECOVERY/REVIEWER_FIVE_QUESTIONS.md

每完成一个Revision都更新。

### Q1：与二次关联有什么不同？

完成：

同分数重复Hungarian；

原规则重解；

学习式二次关联；

JEV方案评价；

评分器×求解器受控对照。

分离模型作用和求解器作用。

### Q2：与固定、可学习阈值有什么不同？

必须用当前最强Pair Threshold作为正式强基线。

重点对比相同合法身份竞争事件。

确认JEV是否提供了阈值无法获得的具体身份区分信息。

不能只说JEV能够动态控制风险。

### Q3：与可学习融合、MLP、普通Attention有什么不同？

同样证据、同样监督、同样模型初始化、同样关联范围、尽量相同参数与计算预算。

如果MLP同样好，不能归因于JEV。

### Q4：参数、FPS和延迟？

必须报告：

新模块参数；

完整系统注册参数；

活跃参数；

实际计算量；

GPU显存；

Stage2各部分时间；

实际完整scene FPS；

同一GPU下与最强普通控制的相对速度变化。

当前参考JEV新增169092参数，3.703FPS。

不得把4B视觉模型的离线吞吐用来证明当前JEV实时性。

### Q5：是不是真正的JEV？

必须明确具有：

Shared State；

真实不同的Typed Questions；

动态合法身份及关联决策空间；

问题条件化证据读取；

可验证的问题输出；

实际联合身份决策；

共享计算复用。

但不能仅凭具有这些组件就宣称JEV独立架构优势。

必须通过动态/固定问题、单问题/双问题、普通Attention、MLP及共享/重复编码消融进行验证。

如果核心能力可以被单一普通阈值完全重现，则不应主张JEV必要性。

---

## 13. 真实在线与安全验收

正式结果必须是完整native closed-loop。

当前检测
→ ReID/VFCE
→ 历史状态
→ 身份决策
→ Hungarian或合法方案执行
→ Bank恢复/START_NEW
→ Gallery与Global ID提交
→ 下一帧真实状态更新。

不能使用teacher-forced正确身份、GT未来动作或原始轨迹的缓存未来关联。

所有替代关联必须满足原生合法约束。

完整报告：

HOTA；
DetA；
AssA；
IDF1；
MOTA；
IDSW；
Frag；
CVIDF1；
CVMA。

专项报告：

跨摄像头错配；
错误合并；
不必要的新ID；
正确身份保持；
真实N01/N10；
过去ID历史检索Precision/Recall；
身份污染持续时间；
KNOWN/UNKNOWN分开统计。

同状态局部反事实与完整冷启动结果分开。

对于多个检测的联合作用，不能将独立边的局部收益直接相加当成全局结果。

Phase XVI历史记忆崩溃案例必须作为回归检查。

不得再次默认启用hard quarantine或未经资格验证的Memory写入控制。

---

## 14. 科学失败之后应该怎样继续

非常重要：

**不得因为第一版HOTA不提升就直接把整个任务标为NO_GO并结束。**

但也不得无限调试直到DEV显示正数。

允许以下自主转向：

如果可认证合法正确身份不在4方案范围内：
先改进GT-free合法方案覆盖，并对普通控制同步更新。

如果正确身份在方案内但JEV排名错误：
分析问题监督和评分器，优先Revision 2。

如果两个Question的答案几乎完全相同：
重新评估其独立语义，必要时改成具有不同真实监督的问题，而不是继续训练相同目标。

如果JEV比普通MLP差：
检查信息是否经过平均融合丢失、有效梯度是否不足、非加性决策是否必要。

如果只在弱Phase XIV基座有效：
执行Revision 3，将有效部分迁移到强Phase XIII Fixed。

如果JEV跨摄像头检索质量较好，但HOTA较差：
分析是身份碎片、错误关联、召回覆盖还是基线差异，而不是单独选择有利指标。

如果所有模型在当前输入下均无法解决关键错误：
审计视觉特征、合法身份可达性与原始GMT 1152D证据差异，报告信息上限。

如果三条Revision全部没有独立收益：
停止盲目扩模型，保留全部负结果，判断是否应转向更合适的结构或投稿周期。

不允许：

扩大已封存TEST；
把UNKNOWN当负例；
将旧DEV调参包装为独立验证；
不报告失败seed；
假造原生因果分支；
给未运行实验填写数字；
通过无限重训赌运气；
修改旧Frozen checkpoint。

---

## 15. GitHub必须持续推送

每完成一个实质阶段：

运行代码测试；

保存实际配置和SHA；

保存真实指标；

保存原始失败；

更新MASTER GOAL；

提交Git；

推送远端；

确认remote HEAD与local HEAD一致；

更新NEXT_ACTION。

不得只在全部研究结束时推送一次。

新分支：

jev/www-jev-recovery-20261012

新目录：

docs/JEV_RECOVERY/

reports/JEV_RECOVERY/

最终至少包括：

MASTER_GOAL.md

QUESTION_DISAGREEMENT_ANALYSIS.md

LEGAL_ASSOCIATION_COVERAGE.md

SUPERVISION_REDESIGN.md

MODEL_REVISIONS.md

PHASE13_STRONG_BASE_EXPERIMENT.md

CROSS_CAMERA_IDENTITY_ANALYSIS.md

REVIEWER_FIVE_QUESTIONS.md

FINAL_RESEARCH_REPORT.md

以及对应：

EXPERIMENT_REGISTRY.json

HYPOTHESIS_REGISTRY.json

TRAINING_RESULTS.json

FAIR_BASELINE_RESULTS.json

FULL_ONLINE_RESULTS.json

N01_N10_ANALYSIS.json

CROSS_CAMERA_RETRIEVAL.json

PARAMETERS_AND_FPS.json

ABLATIONS.json

GO_NO_GO.json

NEXT_ACTION.json

ARTIFACT_SHA_MANIFEST.json

大型checkpoint、视频、原生状态、轨迹与缓存留在服务器。

GitHub上传必要源码、指标、配置、复现命令和简要报告。

---

## 16. 本次任务的完成条件

不要把下面这些当成完成：

检查了数据；
运行了测试；
发现两个Question相关性较高；
训练了三个种子；
完成了某次消融；
生成了NO_GO报告；
完成了Phase的全部脚本。

这些都只是研究过程。

必须按照以下流程持续推进：

**第一步：找到当前JEV失败的结构和监督原因。**

**第二步：修改训练目标、合法关联覆盖或Question读取方式。**

**第三步：实际训练可运行的新JEV。**

**第四步：与最强阈值、MLP、Attention进行严格公平比较。**

**第五步：如果仍未达到预注册目标，进入下一条有充分依据的修改路线。**

**第六步：优先尝试将有效机制迁移至强Phase XIII Fixed基座。**

**第七步：完成真实多摄像头在线指标、身份检索分析和真实FPS。**

**第八步：形成正向证据或严格负结果，并解释JEV的结构价值是否真正成立。**

本次研究允许在已经约定的数据、GPU与有限模型版本预算内自主完成各阶段，不需要每个普通工程步骤都要求用户确认。

但不得把未授权数据范围、独立TEST解封或无限新增大规模计算视为已经获得授权。

最终成功状态必须由真实科研证据决定。

### 现在立即执行的优先顺序

P0：检查当前正在运行的原始任务，保留其结果，创建新分支。

P1：完成Question Disagreement与有效监督统计。

P2：完成合法关联可达性及方案覆盖统计。

P3：完成Revision 1，在相同方案下复核阈值、MLP、Attention和JEV。

P4：完成Revision 2，真正区分跨摄像头一致性与历史稳定性问题。

P5：条件满足时完成Revision 3，把有效机制迁移至Phase XIII Fixed强基座。

P6：完成正式三种子与完整native在线评测。

P7：完成多摄像头身份索引检索、长期错误传播和真实FPS专项评价。

P8：回答五个审稿人问题，提交GitHub全部必要证据。

**最终目标不是得到一个名字叫JEV的模型，而是得到一个确实利用JEV决策结构、能够改善真实多摄像头身份跟踪，并且其额外计算代价值得付出的模型。**

如果科学证据不支持，也必须准确报告失败原因与最强普通替代方法。

不要提前终止在第一轮NO_GO。

不要为了成功降低标准。

现在开始执行。

START NOW.