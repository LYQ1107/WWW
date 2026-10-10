# Phase XV Training and Ablations

三版都完成独立 Tiny128 和 Pilot1500，seed20261009，matching Phase XIV20k 初始化，固定 LAST。Pilot 不接着 Tiny 训练。固定 AdamW lr0.0003/wd0.01/batch4，权重 WHO1/Availability0.5/Trust0.5/Commitment1/whole-payload Assignment0.2/Risk0.5。

| 版本 | Pilot updates | 保留区安全延续准确率 | 必要纠错准确率 | 必要纠错 support | 不安全 horizon 事件 | 资格 |
|---|---:|---:|---:|---:|---:|---|
| 1 | 1500 | 0.9994487320837927 | None | 0 | 13 | NO_GO |
| 2 | 1500 | 0.9989071038251366 | 1.0 | 5 | 18 | NO_GO |
| 3 | 1500 | 0.9982748706152962 | 0.7142857142857143 | 7 | 9 | NO_GO |

V1 实现 Q4 和可靠性，但原保留区没有必要纠错认证；认证准确率不代表所有查询。V2 增加 actual recent3-frame 的局部不安全承诺认证，全球混合 WHO 标签仍 UNKNOWN；新增训练12、保留5个纠错标签。V3 在 V2 原生错误和状态变化证据后，采集它自己真实产生的完整 TRAIN 历史，与原 Multi corpus 混合，固定相同1500 updates，不改变层数、学习率或损失权重。

三版 Tiny 的固定四个代表 payload 均来自原 frozen-Multi corpus，V1和V3相同输入/初始化可产生相同loss与权重；不能把它们当作独立数据复现。V3的新 own-state 干预发生在 Pilot；1500 steps共6000个采样payload中2958来自实际own-v2状态、3042来自原Multi，7个审计纠错样本最终做对5个。Tiny loss下降只证明局部梯度可学，不证明新状态泛化或原生安全。

另保留旧工程 Tiny128：旧联合损失丢掉未知检测行，修复后在新命名空间重跑。总计新梯度 updates5012，其中正式20k=0；旧文件、旧权重和失败日志仍在服务器。训练数据仅12/13/14/16，reserved temporal block不送入优化器，DEV17/18/19仅作诊断，20/21/22封存。

A–F 三种子20k、公平架构消融和 matched4k 正式 on-policy 都依赖可靠 F Pilot。三版都未通过原生安全门，因此这些条件阶段 NOT_RUN、指标 null。历史 Fixed/Multi/Set 仅作历史参考，不能冒充本次同监督/同算力对照；本次没有证明普通网络与 Jev 等价或更优，也没有证明 Jev 独立优势。

反馈通道归因：V1 same checkpoint/prefix/RNG 的24个实际 native 窗口屏蔽 purity/posterior-available 输入，实际 ID 变化数0。该阴性归因只排除该通道对这些V1窗口的动作影响，不外推到后两版权重、其他状态或完整视频；不能将“分布上未覆盖”推断成已证明的因果原因。三版各自 all-query audit、全部不安全 horizon 和 raw轨迹均由 JSON SHA绑定。
