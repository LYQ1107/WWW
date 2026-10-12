# 五个审稿人问题：当前可核查答案

工作仍在执行，未完成项不视为通过。

**Q1：是否只是二次关联？**

重复相同输入 Hungarian 必须保持分配；规则屏蔽边后重解是不同干预。真实评分×求解对照运行中。固定同状态 H64：parent181、JEV173、MLP205、Attention173 个已知错误观测；JEV与Attention相同。精确触发预算诊断单独保存，不替代完整在线结果。

证据：`reports/JEV_ONEWEEK/PAIRED_NATIVE_RESULTS.json / SAME_STATE_BUDGET_DIAGNOSTIC.json`。

**Q2：阈值为什么不够？**

固定、state/pair可学习门限正在同条件比较。新Binary与pair门限是同权重精确重参数化，不能冒充旧Phase VI独立Binary。严格同接口旧VI模型尚未重放。没有证据证明所有阈值都不够。

证据：`reports/JEV_ONEWEEK/MAIN_RESULTS.json / MECHANISM_jev_pilot_SEED20261009.json`。

**Q3：MLP或普通Attention为什么不够？**

新JEV与最强已完成普通控制 threshold 的 HOTA 均值差 -0.377，尚不支持独立架构优势。 各新增控制共享相同冻结XIV初始化、证据、当前标签、最多4方案和4000更新。新模块JEV169092、MLP154980、Attention168964参数；不是整个历史预训练架构的独立公平比较。

证据：`reports/JEV_ONEWEEK/FORMAL_CURRENT_RESULTS.json / NEW_MODULE_PARAMETER_READS.json`。

**Q4：参数和延迟？**

完整原GMT53515309参数；新JEV36165558，总变化−17349751，但完整前端不同，不能把该差值解释为纯关联模块开销。共同前端parent35996466，新JEV增169092。完整视频3次重启FPS和单独组件profile仍以实际完成状态为准；MAC/FLOPs含自定义核缺口，明确PARTIAL。

证据：`reports/JEV_ONEWEEK/PARAMETERS_AND_FPS.json`。

**Q5：是否真是JEV？**

每摄像头因果关联事件共享State编码1次，实际重复编码2次同输出；动态合法ID/私有DEFER和问题条件化均实现。真实TRAIN已知两问题有270个分歧，其中106个旧暴露reserved边JEV同时答对79个、MLP和pair阈值105个、Attention99个；函数可以输出不同语义答案，普通控制同样能做到且表现更好。TRAIN16共同可认证两问题标签分歧0；没有独立JEV语义增益证据。目前应称JEV-inspired关联研究实现，不声称全scene只编码一次、整个生命周期State或TypeSafe商业内部架构复现。

证据：`reports/JEV_ONEWEEK/TRAINED_CONTRACT_CALIBRATION_SEED20261009.json / QUESTION_SEPARATION_AND_PLAN_COVERAGE.json / TYPED_QUESTION_ORDINARY_CONTROL_COMPARISON.json`。
