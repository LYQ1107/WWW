# Phase XXI — 既有 Visual JEV 执行一致性归因

本轮仅复用Phase XVII的12个图像/问题组与两种既有权重，完成三个事前冻结执行配置的72个group-variant、288个路径比较。模型和样本不变，没有新数据、训练或原生动作。计算路径推荐为 **FP32_MATH**；这只是有限样本的数值一致性资格，身份价值、选项顺序鲁棒性与native部署须分别评价。

## 原问题与固定对照

Phase XVII的FP16审计保留原始NO_GO：72个路径比较中22个失败，但决策全部一致，主要失败条件是有效选项raw logit绝对差超过0.05。本轮保持logit≤0.05、概率差≤0.02、决策完全一致的原门槛，比较FP16默认SDPA、FP16数学SDPA、FP32数学SDPA。默认→数学只改变后端；数学FP16→FP32只改变执行精度。没有逐算子kernel trace，因此不把默认配置说成所有算子均使用某个特定kernel。

每组均验证真实完整processor token、公共前缀位置一致及KV fork无别名；图像来自同一原始pair/哈希。五个问题为WHO、OWNER_EVIDENCE、AMBIGUITY、CROSS_VIEW、FALSE_PREMISE。两种权重是既有Qwen3-VL4B与公布answer LoRA，保留原LM head，不训练新任务头。原始24组中 **24/24** 的三个旧路径和顺序判定精确复现。

| 配置 | vision-cache PASS /24 | batch PASS /24 | prefix-share PASS /24 | prefix-share-batch PASS /24 | 最大logit差 | WHO反序语义决策一致 |
|---|---:|---:|---:|---:|---:|---:|
| FP16_DEFAULT | 24 | 18 | 15 | 17 | 0.21875 | 18/24 |
| FP16_MATH | 24 | 14 | 16 | 17 | 0.125 | 18/24 |
| FP32_MATH | 24 | 24 | 24 | 24 | 0.00018310547 | 18/24 |

在这批固定样本上，两个FP16配置均未全部通过，FP32数学配置全部通过。这支持将此前的数值门槛失败归因到低精度执行，而非另一个token、位置或KV语义修复；有限样本不证明全局等价，也不说明所有数值差异仅来自某一种kernel。完整前向的跨配置决策变化及概率差记录在VISUAL_PROFILE_ATTRIBUTION.json。相比默认FP16，FP16数学与FP32数学的120个完整前向决策均无变化，但最大概率差分别约0.0618和0.0562，因而不能将旧配置的confidence校准自动视为新配置已验证。这些跨配置量不能由缓存对照中决策一致直接推出。所有配置完整披露，不按输出换样本、调容差或保留有利结果。

选项反序会改变实际提示词；即使token/cache完全正确，模型仍可能改变同一身份选项的判断。这是候选表达的敏感性，不是相同提示词的KV计算错误。数值一致性通过也不升级此前教师身份识别价值或native候选控制器资格。有限12组不是独立新验证集。

## 实际成本

每个配置/权重/N/路径先warmup一次，再计两次GPU同步执行。报告总batch延迟、两次原值/中位数/最大值、峰值显存；prefill与KV fork单独计时。两次测量不能估计p95，p95明确为null。N32为同一已解码蒙太奇的重复问题压力测试，不是32个真实动态身份候选，也不是在线MOT吞吐或包含图像IO的服务延迟。

| 配置 | 权重 | N32串行总ms | batch总ms | 共享总ms | 串行/共享 | batch/共享 |
|---|---|---:|---:|---:|---:|---:|
| FP16_DEFAULT | pretrained | 2856.2 | 975.5 | 446.6 | 6.40 | 2.18 |
| FP16_DEFAULT | published_answer_LoRA | 4196.8 | 1377.9 | 635.3 | 6.61 | 2.17 |
| FP32_MATH | pretrained | 6338.7 | 4419.7 | 1680.1 | 3.77 | 2.63 |
| FP32_MATH | published_answer_LoRA | 6823.1 | 4690.1 | 1787.7 | 3.82 | 2.62 |

## 工程失败与复现

第一次在完成FP16_DEFAULT的24组与全部计时后，下一配置准入失败：模型仍由最后的bound method及parameter引用保留，显存未释放。原FAILURE、部分结果和原提交5084c075完整保留。唯一普通工程修复在b1b78b40：显式释放model、method、parameter，保留所有科学设置，在独立v2目录重跑。三个配置释放后均实测free memory≥26,000MiB；少量剩余allocated memory按原值记录，不声称为零。旧结果不覆盖。不是架构科研失败，也没有放宽26,000MiB准入要求。

成功actor固定于b1b78b40307928c6737035c0ebd9d1b6a8b38b56，源码SHA与该Git提交逐项核验；协议SHA 2f0eed5c8b68c36166b29fe6e3d788729da6e71c0693551819caa09bb527a5a7。三份权重文件SHA实测一致；Phase XX全部27个发布文件及Phase XVII原审计不变。单GPU2，成功重跑约508.5秒，两轮留存约0.444MiB，home空闲37.00GiB。失败日志、精简JSON与代码推GitHub，图像/模型留服务器。

## 主科研目标

本轮解决或限定的是执行一致性，未建立新的身份识别、全视频关联收益或架构独立优势。现有安全监督仍为3TRAIN＋1预留，低于8＋3门槛；训练NOT_RUN。原生cache服务没有在此轮实现或验证，冷启动、跨状态失效、活跃图库与真实图像流水线均不能借这次microbenchmark宣称完成。论文仍NOT_READY，MASTER继续ACTIVE/PIVOT_REQUIRED。

下一步沿已推送Phase XX独立来源方案进行数据资格审计，等待新增来源范围/合法数据路径回复；本轮没有读取新增数据、sealed20/21/22或官方TEST。复核入口：PREREGISTRATION、VISUAL_EXECUTION_PROFILE_PARITY、VISUAL_PROFILE_ATTRIBUTION、VISUAL_EXECUTION_LATENCY、ENGINEERING_AND_DATA_AUDIT、FINAL_GO_NO_GO及ARTIFACT_SHA_MANIFEST。
