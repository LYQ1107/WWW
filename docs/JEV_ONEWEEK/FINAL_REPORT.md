# JEVTrack 本周实验报告

当前结论：JEV独立贡献 **NOT_ESTABLISHED**。此报告仍是进行中的证据汇总；正式对照、归因、消融及完整视频效率未全部完成前不宣布任务完成。

| 方法 | 完成种子 | HOTA ± seed SD | AssA | CVIDF1 | CVMA | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XIII_fixed_reference | 3 | 75.637 ± 0.379 | 77.907 | 91.889 | 87.888 | 105.0 |
| XV_V3_reference | 1 | 74.188（单种子） | 74.891 | 90.286 | 86.781 | 147.0 |
| additive | 3 | 73.383 ± 0.551 | 73.365 | 87.915 | 86.724 | 133.0 |
| attention | 3 | 72.870 ± 1.918 | 72.383 | 86.828 | 86.752 | 125.0 |
| base_repeat | 1 | 73.476（单种子） | 73.527 | 87.196 | 86.523 | 824.0 |
| base_rule | 1 | 73.476（单种子） | 73.527 | 87.196 | 86.523 | 824.0 |
| base_single | 1 | 73.476（单种子） | 73.527 | 87.196 | 86.523 | 824.0 |
| cosine | 1 | 69.085（单种子） | 65.052 | 64.843 | 46.450 | 82.0 |
| cross_only | 3 | 73.018 ± 0.856 | 72.673 | 87.181 | 86.378 | 133.7 |
| deepsets | 3 | 73.272 ± 1.604 | 73.142 | 87.405 | 86.386 | 141.7 |
| extra_hungarian | 3 | 73.505 ± 0.856 | 73.639 | 88.546 | 87.595 | 391.7 |
| fixed_fusion | 3 | 73.352 ± 0.621 | 73.380 | 88.154 | 87.680 | 288.3 |
| fixed_question | 3 | 72.909 ± 0.727 | 72.407 | 86.543 | 86.094 | 138.7 |
| fixed_threshold | 3 | 73.514 ± 0.838 | 73.658 | 88.579 | 87.598 | 395.7 |
| jev | 3 | 73.374 ± 0.720 | 73.360 | 88.092 | 86.587 | 123.3 |
| linear | 3 | 73.357 ± 0.451 | 73.378 | 87.475 | 87.160 | 327.0 |
| mlp | 3 | 73.493 ± 0.997 | 73.582 | 87.286 | 85.930 | 114.0 |
| no_cross_evidence | 3 | 66.495 ± 6.718 | 60.649 | 56.496 | 36.520 | 716.0 |
| no_question_reader | 1 | 72.454（单种子） | 71.536 | 84.052 | 85.633 | 134.0 |
| no_stability_evidence | 3 | 59.083 ± 0.949 | 47.753 | 61.991 | 83.192 | 9423.0 |
| parent | 3 | 73.505 ± 0.856 | 73.639 | 88.546 | 87.595 | 391.7 |
| stability_only | 3 | 73.567 ± 0.651 | 73.748 | 88.088 | 86.540 | 125.0 |
| state_threshold | 2 | 71.216 ± 0.219 | 69.144 | 82.898 | 85.089 | 548.5 |
| threshold | 3 | 73.752 ± 1.873 | 74.134 | 87.795 | 85.752 | 115.7 |

上述学习模型首先对DEV17/18/19六个相机进行真实pooled评测，再计算三个固定种子的均值/样本标准差；种子波动不是三个独立数据集或显著性证明。DetA/IDF1/MOTA/Frag和逐种子数据见MAIN_RESULTS。全部主指标为未做未来短轨筛选的完整在线结果，CVIDF1/CVMA来自冻结官方MATLAB。

**1. 原GMT与已有/新JEV达到什么水平？**

原始完整GMT冻结参考HOTA78.110；前端1152D/Stage2权重不同。共同前端XIV parent三种子HOTA73.505、AssA73.639、CVIDF1 88.546；新版和旧参考见主表，不能跨训练阶段拼最佳值。

实验/具体数值：上方完整主表及 `MAIN_RESULTS.json / BASELINE_PARITY.md`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**2. 新模型是否提高多摄像头Global ID一致性？**

新JEV在已完成三种子上未全面提高Global ID一致性：检索精确率、召回、原始ID保持和错误传播必须共同判断。

实验/具体数值：上方完整主表及 `CROSS_CAMERA_ANALYSIS.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**3. 哪些错误被真正纠正？**

N01记录实际同检测行从owner错误变为正确，但完整独立轨迹的ID namespace不同，不证明原始ID被恢复；详细与同状态H64结果分开。

实验/具体数值：上方完整主表及 `CROSS_CAMERA_ANALYSIS.json / PAIRED_NATIVE_RESULTS.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**4. 哪些正确关联被反改错？**

N10真实反改错完整保留，不能只报告IDSW下降。见CROSS_CAMERA_ANALYSIS里的各方法计数、UNKNOWN与右删失。

实验/具体数值：上方完整主表及 `CROSS_CAMERA_ANALYSIS.json / SAME_STATE_BUDGET_DIAGNOSTIC.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**5. 比简单二次关联多了什么？**

重复相同输入 Hungarian 必须保持分配；规则屏蔽边后重解是不同干预。真实评分×求解对照运行中。固定同状态 H64：parent181、JEV173、MLP205、Attention173 个已知错误观测；JEV与Attention相同。精确触发预算诊断单独保存，不替代完整在线结果。

实验/具体数值：上方完整主表及 `MAIN_RESULTS.json / PAIRED_NATIVE_RESULTS.json / SAME_STATE_BUDGET_DIAGNOSTIC.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**6. 比固定/学习式阈值多了什么？**

固定、state/pair可学习门限正在同条件比较。新Binary与pair门限是同权重精确重参数化，不能冒充旧Phase VI独立Binary。严格同接口旧VI模型尚未重放。没有证据证明所有阈值都不够。

实验/具体数值：上方完整主表及 `MAIN_RESULTS.json / MECHANISM_jev_pilot_SEED20261009.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**7. 比融合、MLP和普通Attention多了什么？**

新JEV与最强已完成普通控制 threshold 的 HOTA 均值差 -0.377，尚不支持独立架构优势。 各新增控制共享相同冻结XIV初始化、证据、当前标签、最多4方案和4000更新。新模块JEV169092、MLP154980、Attention168964参数；不是整个历史预训练架构的独立公平比较。

实验/具体数值：上方完整主表及 `MAIN_RESULTS.json / TYPED_QUESTION_ORDINARY_CONTROL_COMPARISON.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**8. 两个问题为何需要共享状态，能否退化为普通Head？**

共享状态是一项真实计算复用，不是已证实的准确率创新。两问题在共同已知TRAIN16目标完全一致，独立语义增益尚未成立。

实验/具体数值：上方完整主表及 `TRAINED_CONTRACT_CALIBRATION_SEED20261009.json / ABLATIONS.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**9. 实现是否符合JEV思想？**

每摄像头因果关联事件共享State编码1次，实际重复编码2次同输出；动态合法ID/私有DEFER和问题条件化均实现。真实TRAIN已知两问题有270个分歧，其中106个旧暴露reserved边JEV同时答对79个、MLP和pair阈值105个、Attention99个；函数可以输出不同语义答案，普通控制同样能做到且表现更好。TRAIN16共同可认证两问题标签分歧0；没有独立JEV语义增益证据。目前应称JEV-inspired关联研究实现，不声称全scene只编码一次、整个生命周期State或TypeSafe商业内部架构复现。

实验/具体数值：上方完整主表及 `MODEL_AND_NATIVE_CONTRACT_TESTS.json / QUESTION_SEPARATION_AND_PLAN_COVERAGE.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**10. 参数、FPS、Stage2与显存是多少？**

完整原GMT53515309参数；新JEV36165558，总变化−17349751，但完整前端不同，不能把该差值解释为纯关联模块开销。共同前端parent35996466，新JEV增169092。完整视频3次重启FPS和单独组件profile仍以实际完成状态为准；MAC/FLOPs含自定义核缺口，明确PARTIAL。

实验/具体数值：上方完整主表及 `PARAMETERS_AND_FPS.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**11. 是否提高HOTA/AssA/CVIDF1并保持身份稳定？**

新JEV主实验已为真实完全mutated-state online，所有检测/ID提交和Gallery/Bank会影响下一帧；主要HOTA/AssA相对MLP未达到预注册探索目标，CVIDF1优势不一致。

实验/具体数值：上方完整主表及 `MAIN_RESULTS.json / CROSS_CAMERA_ANALYSIS.json`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

**12. 是否足以支撑WWW独立贡献论文？**

目前不足以支撑声称JEV具有明确独立贡献的WWW论文。持续更新多摄像头身份索引与错误检索属于真实Web相关问题；研究动机不代替独立性能与可靠性证据。

实验/具体数值：上方完整主表及 `GO_NO_GO.json / REVIEWER_FIVE_QUESTIONS.md`，未完成项为NOT_RUN/PARTIAL。模型/权重：该工件cases中的seed/checkpoint/initializer路径与SHA，新增模块均为最后4000-update权重，旧参考另标训练阶段。复现：`REPRODUCTION.md` 中对应CLI，服务器运行根 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1`。局限：已暴露TRAIN/DEV，完整GMT不同前端；本项不证明独立安全或未来价值。证据状态：实际工件的COMPLETE/PARTIAL/NOT_RUN；独立JEV贡献NOT_ESTABLISHED。

复现代码、协议、START和权重SHA均绑定每个工件。模型保存在 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1/training_v1/`，真实轨迹/详细事件留服务器；GitHub只传源代码、小指标、失败、配置与SHA清单。参数完整前向读数仅覆盖真实TRAIN12首2帧，未观察Bank分支不能当成永久Dormant。

关键局限：新梯度只有TRAIN12/13/14非旧reserved时间块，TRAIN16与DEV均已旧暴露；前端旧Stage1训练曾见全部TRAIN。统一TRAIN注释文件包含全部TRAIN行，助手只筛选允许视频的GT用于本轮监督/评测；20/21/22不参与本轮模型输入、训练或评测。未读取官方TEST，不启动Full24，没有新下载。当前GT认证的困难选项多数UNKNOWN，置信度校准是预暴露控制器诊断而非独立泛化。完整原GMT前端和主表共同Stage1不同；M5是XIV/native规则，未冒充原完整GMT传统cascade；旧VI Binary接口不一致。原MASTER长期安全Goal及8TRAIN/3RES安全门均未通过、未完成。

状态与原始工件：`EXPERIMENT_REGISTRY.json`、`MAIN_RESULTS.json`、`ABLATIONS.json`、`PARAMETERS_AND_FPS.json`、`CROSS_CAMERA_ANALYSIS.json`、`GO_NO_GO.json`、`ENGINEERING_FAILURES.json`、`ARTIFACT_SHA_MANIFEST.json`。缺失实验均为NOT_RUN/PARTIAL，不以零代替。下一步只完成冻结有限矩阵，不通过扩大模型、反复调DEV或挑选种子制造成功。
