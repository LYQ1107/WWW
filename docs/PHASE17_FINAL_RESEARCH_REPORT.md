# Phase XVII 最终研究报告

本轮诊断已完成，MASTER保持 **PIVOT_REQUIRED**，独立JEV增量价值 **NOT_ESTABLISHED**，论文 **NOT_READY**。冻结参考为666e2b922eb676b3552db0c6dc8b67b1b3978450；未新增身份任务训练，未覆盖checkpoint或旧结果，535个既有引用（包含339个实际权重文件）重新校验通过。20/21/22仍封存、official TEST未读取、Full24未启动。Stage1历史接触全部24个TRAIN已披露。

原生2×2实验的actor为3958c80bbf203ec0a1bf13543d927d98db7b50e7，固定XV V3、真实Stage1 ROI/ReID、合法候选生成/求解规则与起始状态，各分支独立执行真实mutated-state未来。B0及B3完整视频指标精确复现XV/XVI端点，24组前缀B0/B3的H32 ID提交也全部精确匹配旧结果，不能把本轮解释为新模型训练优劣。

|分组|HOTA|AssA|IDF1|IDSW|MOTA|CVIDF1|CVMA|
|---|---:|---:|---:|---:|---:|---:|---:|
|B0|74.1883|74.8907|90.5210|147.0|89.6594|90.2859|86.7815|
|B1|65.8145|59.2444|75.5301|267.0|89.3477|66.9981|68.2820|
|B2|73.5722|73.7100|89.8437|330.0|89.2045|89.1459|85.4367|
|B3|61.7644|52.0522|71.0264|1494.0|86.2586|67.4761|72.5801|

IDSW归因：WHO单独+120、Q4单独+183，联合+1347；交互差分+1044。交互是代数因果对比，不能唯一指定为某一个内部操作的贡献。B1的跨相机CVIDF1=66.9981，比单看267次IDSW显示的问题更严重。首次实际ID分歧为DEV17 WHO/联合frame0、Q4 frame38；DEV18 frame280/488；DEV19 frame3/106。冷启动fallback只保留原始向量但历史槽位/元数据不全，滞后Q4与原生提交相互作用；frame0 owner未知，ID改变不是自动的错误merge证据。

96个前缀分支的原生/raw-memory/RNG起点相同。H8/16/32严格有害检查B1/B2/B3为15/3/17，局部收益组1/2/3；这些是重叠窗口，不能计独立样本。B3第三个局部组在旧XVI H64已混合+3、假拆分+1，不能成为长期安全标签；原XVI18次有害检查及2个安全组结论完整保留。错误传播按检测可观测owner错误连续帧计，缺失观测分段并右删失，UNKNOWN不当错或恢复。

273个严格TRAIN首payload事件中，只有1个所选owner错误且合法正确owner候选可用的机会；不等同已经找到安全联合纠错动作。187个候选池超过16。12个困难组、261个全部合法候选进入离线teacher，不Top16、不比较独立softmax作全局排名。组按预注册video/owner/camera/time-block划分，跨相机同owner/time仍可能相关，未作独立样本显著性声明。C4只是余弦间隔代理，不能据此断言信息理论上不可辨识。

无新增head的Qwen预训练及公开通用QA LoRA的WHO准确率49.17%/28.93%，固定余弦探针95.45%；唯一机会中正确ID2与错误ID6都被视觉版本选为same。通用QA适配还失去假前提拒答能力。真实完整分词检查发现上游prefix/suffix拼接缺陷，保留原失败、在局部适配器中修复并用serial full forward重跑；新token全部验证通过。FP16数值parity或选项逆序仍失败，语义决定一致仅18/24组。原LM head无需新增head，但使用已训练的通用QA LoRA不能叫“不需要训练”。

E1正权重时间衰减、E2摄像头平衡均值各执行24个原生前缀，actor6224eab；原始观测、最新槽位、原始Q4、metadata和合法动作全部保留。两者分别13/4次严格有害H检查，0个局部安全收益组，按事前门槛完整冷启动NOT_RUN。未调参重试，未把正权重当owner认证。它们有更少局部伤害，但没有完整视频整体优于hard的证据。

基于实际原生提交的身份索引因果回放，跨相机query precision B0/B1/B2/B3=91.76%/60.14%/90.82%/62.80%，身份连续性95.04%/76.85%/95.17%/84.85%。索引先查询过去其他相机观测再写当前，键为实际预测GlobalID；GT仅在独立评价表中使用。只覆盖detector-observed语料、没有bootstrap-only行，不是Web系统部署或新独立数据集。共享teacher的N32离线诊断加速6.41x/6.73x不计合格在线收益；三次计时的p95为最大值，N>5重复问题。旧真实图像profile约3.25场景FPS、Stage2p95约56ms，仅256场景帧，不是完整视频未插桩FPS；新teacher未计入在线路径，部署目标未通过。

Tiny、pilot、20k、matched4k、身份任务LoRA、新head、同图像MLP/DeepSets/Attention与三生命周期架构比较全部NOT_RUN，指标null。数据/原生安全/公平对照前置门槛不满足，不能通过解封DEV/TEST或额外训练制造结果。

|问题|回答|
|---|---|
|Q1|冻结XV权重下，hard quarantine同时改变WHO/Q4；正确视角观测被排除、bootstrap fallback缺少历史槽位/元数据、Q4前驱滞后，原生反馈放大错误。不是新模型训练失败。|
|Q10|必要性尚未证明。共享状态存在因果交互，但三生命周期共享模型与分开模型的公平比较未运行。|
|Q11|尚不具备就绪的WWW增益论文证据。可复现负向机制归因与因果身份索引诊断有价值，但独立泛化、强基线、Web任务和部署门槛仍缺失。|
|Q12|优先从授权TRAIN完整原生轨迹补充独立owner冲突/合法候选事件，并执行固定候选干预的真实未来，判断是候选/证据不足还是联合动作错误；不扩大VLM、不解封heldout。下一轮先冻结协议和预算。|
|Q2|IDSW147→WHO单独267(+120)、Q4单独330(+183)、联合1494(+1347)，交互差分+1044；仅为冻结系统四组因果对照，不能唯一分配交互项的物理原因。|
|Q3|会。相同起始状态、96个真实未来分支中WHO/Q4/联合有15/3/17个有害H检查；短期局部改善不抵消长期伤害，已知H64反转保留。|
|Q4|部分过去owner可判定，但273个严格事件只发现1个所选owner错误且合法正确候选存在的机会；12个困难组不足以形成8+3安全纠错监督。C4数值代理不是信息不可能证明。|
|Q5|本轮没有合格新增owner证据。WHO准确率49.17%/28.93%，固定余弦探针95.45%；唯一机会中两视觉版本把正确ID2和错误ID6均判为same。图像一致性不等于所有权。|
|Q6|N32离线共享执行诊断加速6.41x/6.73x，单问14.12/20.10ms，N1更慢；数值parity或语义选项顺序资格失败，非在线MOT吞吐收益。仅3次计时，N>5重复问题，K压力测试非多ID排名证明。|
|Q7|两条软规则均失败：E1有13、E2有4个严格有害H检查，0个局部安全收益组；相比hard联合17次局部伤害少，但不能宣称完整视频整体优于hard。完整冷启动按事前门槛NOT_RUN。|
|Q8|尚不能判断。公平同证据MLP/Attention训练NOT_RUN；WHO/Q4对照和余弦探针不是架构消融，无独特结构增益结论。|
|Q9|未证明。反复使用DEV、teacher和软规则不合格；20/21/22仍封存，没有独立验证或显著性优越性证据。|

完整状态与来源：FINAL_GO_NO_GO.json、WHO_Q4_FACTORIAL.json、FIRST_DIVERGENCE.json、OWNER_IDENTIFIABILITY.json、VISUAL_JEV_COMPARISON.json、SOFT_MEMORY_CAUSAL_RESULTS.json、IDENTITY_INDEX_RETRIEVAL.json。执行原始目录为/home/liuyeqiang/WWW_jev_phase17_runtime/20261011_v1；全量视频、权重、状态、逐帧gzip日志留服务器，GitHub只推代码、紧凑结果与SHA。独立视觉环境未改变GMT环境。两份从未使用的QA seed权重已清理，释放约252MiB，所用基座/根adapter和原下载manifest保留。

下一轮见PHASE17_NEXT_EVIDENCE_PROTOCOL.md：先冻结授权TRAIN完整轨迹的候选/owner证据采集及合法动作干预预算。当前轮变体预算用尽，不自动训练、不扩大VLM、不解封20/21/22。满足8个独立安全纠错组、3个保留组、反改错负例、完整冷启动和公平控制后才可学习；如当前TRAIN仍不足，先提出额外数据划分与资源需求。MASTER未标科研完成。

复现执行记录（原run_id已有输出，勿覆盖；完整重跑须在隔离新run目录和冻结源码副本下执行）：

```bash
# 工程与先前证据核查
/home/liuyeqiang/anaconda3/envs/GMT/bin/python /home/liuyeqiang/WWW_jev_phase17/reproduction_tools/jev_phase17_integrity.py
# 已执行的冻结actor入口：12冷视频+96前缀
/home/liuyeqiang/anaconda3/envs/GMT/bin/python /home/liuyeqiang/WWW_jev_phase17_runtime/20261011_v1/source_factorial_v1/reproduction_tools/jev_phase17_queue.py
# 已执行的软规则冻结actor入口：48前缀+2契约探针
/home/liuyeqiang/anaconda3/envs/GMT/bin/python /home/liuyeqiang/WWW_jev_phase17_runtime/20261011_v1/source_soft_v1/reproduction_tools/jev_phase17_soft_queue.py
# teacher执行环境与actor见VISUAL_JEV_COMPARISON；实际actor db613a8
# PHASE17_ACTOR_COMMIT=db613a8... CUDA_VISIBLE_DEVICES=0 visual_env/bin/python reproduction_tools/jev_phase17_teacher.py
```

精确模型revision、文件SHA、裁剪来源、包版本、GPU、actor源码SHA、原始结果SHA均在JSON中。校验阶段包括源码编译、真实原生状态/ID parity、WHO/Q4不变项断言、完整tokenization、M-RoPE/KV分叉、执行路径误差、选项逆序、真实GPU推理与未修改的MATLAB/TrackEval。原源码BPE缺陷、FP16/顺序资格失败不是测试PASS；所有科学负结果保留。
