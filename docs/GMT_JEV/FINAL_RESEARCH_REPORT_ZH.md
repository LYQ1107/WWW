# GMT-JEV 当前中文研究报告

当前结论：完整GMT恢复和集成等价已通过；固定300步普通MLP是已测开发质量正向候选。Attention/CIC未通过本轮质量门槛。已测质量、效率、机制、种子和独立资格分别判断，SCIENTIFIC_SUCCESS=false。
本报告由当前JSON生成，所有方法均列出。主要协议固定为native raw；原生min_track_len=50及stock后处理另报。TRAIN16已暴露，所有结果属于开发证据。

## 1. 原始完整GMT基线

冻结原项目04324354源码和完整Stage2 checkpoint，严格加载411个tensor、53,515,309参数。VFCE1024+RPCE128形成1152维GTA输入，检测、GTA Transformer、原生候选/匹配、Gallery/Bank和REACT/WRITE全部保留。旧JEV executor、候选替代器和perception cache均关闭。
TRAIN16为1034场景帧、2068张真实图像；checkpoint SHA256：cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8。Original与Adapter OFF在32帧和完整1034场景的logits、分数、ID提交、Gallery/Bank及预测严格一致，见 BASELINE_PARITY.json。
三个learned gate零残差头另有32场景的严格原生核验；该证据不冒充新模式的整段逐项零残差核验。旧零残差原型的完整stock指纹/效率结果单独保留。

## 2. 可纠正机会与监督资格

完整TRAIN12/13/14的冻结Rule主要HOTA均下降，合并ΔHOTA −0.008859，IDSW477→533；严格同状态接受式参考N01/N10=0/1。旧失败不改写，完整TRAIN结果不替代TRAIN16或独立验证。
受限ORACLE是离线动作空间见证，未执行其动作，不是模型可表达性、最优上界或在线收益证明。手工门控排除了大部分认证错误目标，支持一次统一learned risk mode修订，不支持无限架构搜索。

TRAIN时间桶共51567行，其中26297行有唯一认证合法目标；原GMT错误97行，17行通过阈值margin及联合有界重评分见证。临时缓冲峰值14.559MiB，保留全部17条合格E1和3942条preserve监督行。
GT仅作离线认证；首个历史owner需至少3次已知观测且覆盖率≥80%，年轻别名也会阻止伪造唯一目标。UNKNOWN、目标冲突和有界不可达行不伪造监督，但完整候选矩阵仍参与原生约束训练。
时间留出桶矩阵不进入optimizer，历史状态与认证跨桶连续；较早留出桶GT可参与随后TRAIN标签的历史认证。因此这不是严格标签隔离或独立验证。TRAIN16没有梯度更新。

## 3. 实际模块与训练

模块只在GTA聚合出native Global ID分数后添加残差：S_new=S_GMT+0.05×sigmoid(gate)×tanh(delta)×原支持长度。第一场景跨相机入口和后续普通MATCH均接入；原生候选、求解器、接受阈值和提交规则保留，Bank打分路径不替换。
相同24维可观察输入涵盖GMT分数竞争、历史数量/年龄、同相机与其他相机外观支持、冲突、检测置信度和合法候选信息。派生摘要仅来自已提交的原生ID，不另建ID分配器或独立Gallery。
MLP5119、Attention4898、CIC4962参数，CIC/Attention latent32，MLP hidden43。CIC增加观测冲突角色嵌入；不能因命名就声称JEV结构价值。
三个头均已完成300步，使用相同临时样本、采样顺序、seed20261009、AdamW(lr0.001, weight_decay0.0001)、FP32及clip1；只保存固定300步终点，不按TRAIN16选step。原GMT冻结。Loss为native assignment结构化hinge+接受阈值hinge+10×preserve残差平方+全矩阵残差平方；margin0.005。真实样本梯度检查通过。具体顺序SHA、checkpoint SHA和训练记录见 TRAINING_METRICS.json / LEARNED_GATE_SMOKE_TRAINING.json。

## 4. 完整冷启动主要raw成绩

| 方法 | HOTA | ΔHOTA | AssA | DetA | IDF1 | CVIDF1 | CVMA | MOTA | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GMT | 68.054089 | +0.000000 | 67.064493 | 69.411013 | 87.182280 | 86.544619 | 87.093920 | 87.000434 | 1898 | 503 |
| Threshold | 68.189257 | +0.135169 | 67.179802 | 69.563829 | 87.645192 | 86.867339 | 87.347668 | 87.007112 | 1898 | 499 |
| Rule | 68.556671 | +0.502582 | 67.900740 | 69.558609 | 88.458691 | 87.517847 | 87.237488 | 86.517979 | 2191 | 503 |
| Confidence | 68.315141 | +0.261053 | 67.560472 | 69.421137 | 87.805169 | 86.921407 | 87.372709 | 87.102267 | 1839 | 500 |
| MLP | 68.654661 | +0.600573 | 68.098573 | 69.552818 | 88.649302 | 88.001081 | 87.154018 | 86.928650 | 1945 | 501 |
| Attention | 67.508007 | -0.546082 | 65.804120 | 69.619668 | 86.698946 | 85.595046 | 86.431171 | 86.891923 | 1965 | 497 |
| CIC | 67.365381 | -0.688707 | 65.526387 | 69.620534 | 86.355166 | 85.699803 | 86.816801 | 86.788421 | 2031 | 502 |

MLP的已测raw质量门槛通过，IDSW仍增加2.48%，在预注册≤1.05倍范围内；MOTA下降约0.0718点，不能隐藏。Attention/CIC的HOTA、AssA、CVIDF1下降，CIC的IDSW还超过1.05倍，均未通过。以上为单种子开发差值，不是统计显著性或独立验证。

## 5. 身份稳定性与真实提交

| 头 | 正确接受式参考条数 | 保持率 | N01 | N10 | UNKNOWN目标 | 回顾性不一致相机帧 | 历史锚错误相机帧 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mlp | 30150 | 99.996683% | 0 | 1 | 27705 | 4407 | 3795 |
| attention | 29925 | 99.993317% | 0 | 2 | 27922 | 5908 | 4091 |
| cic | 29728 | 99.989909% | 0 | 2 | 28099 | 5774 | 4700 |

GMT的回顾性不一致持续量为5274相机帧，严格历史锚错误持续量为4054。MLP两者分别为4407和3795。前者依赖整段映射，后者只用更早场景的首个≥3/80%认证owner，并对未知/缺失/视频结束作删失；两者都不是真实单次干预的因果传播秒数。
N01/N10只与同一当前变异状态下原始GMT的接受式MATCH比较，原始拒绝后的Bank反事实未知；不同方法的认证覆盖不同。HOTA正号不能自动解释为这些局部计数的正向。

首次可见ID分歧在frame2/view1/row22。三场景重跑确认，事件前完整native状态/RNG、派生摘要、原始分数、24维输入、候选和支持长度严格一致；GMT实际创建ID52，MLP接受已有ID25。两条前缀与各自完整轨迹一致，证明实际提交有变更。
但该行GT为UNKNOWN，不能计为认证纠错。未执行事件单独开启/关闭的完整视频干预，不能将全视频增益归因于此，也未证明真正错误传播时间。见 LEARNED_FIRST_DIVERGENCE_NATIVE.json。

## 6. 普通对照与结构价值

三头使用相同证据、合法候选、原生求解器、残差上界、数据和训练预算。当前普通MLP优于Attention/CIC，按任务要求保留MLP，不为JEV强行扩大架构。CIC当前fixed300开发方案NO_GO，整个研究目标仍未完成。
这不证明跨相机特征必需，也不排除普通分数/阈值校准解释大部分收益。缺少训练后证据消融、合格种子和独立视频，不能宣称JEV-inspired结构优于普通控制器或已满足WWW创新要求。所有当前修正仍使用同一native solver，历史额外global re-solve结果不并入本研究。

## 7. 真实完整图像效率

| 方法 | 完整图像FPS | 相对GMT | 三轮状态 | 95%门槛 |
|---|---:|---:|---|---|
| GMT | 3.409768 | 1.000000 | COMPLETE | True |
| Threshold | 3.225844 | 0.946059 | COMPLETE | False |
| Rule | 2.993950 | 0.878051 | COMPLETE | False |
| Confidence | 3.392963 | 0.995072 | COMPLETE | True |
| MLP | 3.129765 | 0.917882 | COMPLETE | False |
| Attention | 3.059589 | 0.897301 | COMPLETE | False |
| CIC | 2.396665 | 0.702882 | COMPLETE | False |

同一GPU7/V100-DGXS-32GB，每次重置原生状态、派生摘要和seed。包含真实图像读取/resize、检测、VFCE/RPCE、GTA、原生匹配/Gallery/Bank及最终stock处理；不含模型加载、8帧warmup和GT评测。图像FPS=2×场景FPS。Stage2 p50/p95和峰值显存见各 SPEED_*.json。共享宿主机其他卡负载存在变化，串行块不能独立分离系统负载、状态轨迹和模块开销的因果贡献。
MLP默认平均完整图像FPS为3.129765、相对GMT0.917882，未通过95%，不能把质量正向称为完整目标成功。24场景GPU8组件诊断只有定位意义，不外推全视频FPS。
单次缓存检查条件及严格核验要求见 LEARNED_MLP_CACHE_PROTOCOL.md：等待默认计时结束；预算足够时先核验完整2067次features/gate/residual/score及raw预测严格一致，再单列三轮缓存计时。旧Rule缓存没有观察到加速，其结果不改写。不通过等价性就不测缓存速度。

## 8. 同约定stock次要结果

| 方法 | HOTA | AssA | CVIDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| GMT | 68.389411 | 68.072882 | 87.641219 | 542 | 586 |
| Threshold | 68.541303 | 68.152887 | 87.963891 | 629 | 582 |
| Rule | 68.966066 | 69.235313 | 88.912758 | 411 | 608 |
| Confidence | 68.642726 | 68.488564 | 87.951663 | 637 | 568 |
| MLP | 69.043870 | 69.259067 | 89.264664 | 398 | 554 |
| Attention | 67.862465 | 66.883228 | 86.754797 | 444 | 554 |
| CIC | 67.770719 | 66.714237 | 87.012530 | 393 | 544 |

stock不是主要协议。最终轨迹长度过滤依赖整段视频，不能用其较好结果覆盖raw失败或称作在线已知的因果修复。官方MATLAB toolkit SHA不变，TrackEval/转换器协议冻结；具体原始输出和输入SHA见各ONLINE报告。

## 9. Goal、资格和下一步

MASTER_GOAL状态 ACTIVE，scientific_success=False。完整基线和工程接入已证；MLP已测质量正向，默认效率失败；必要种子、MCMT特征/机制贡献和合格独立完整视频尚未证明。应用Goal不能据此标为科学完成。
数据范围仅TRAIN12/13/14学习、TRAIN16开发；不新增DEV17–19、封存20–22、官方TEST、Full24或新数据下载。TRAIN16不能伪装独立验证，旧阶段NO_GO/checkpoint保留。
先完成已注册默认计时和条件缓存工程检查，不扩大架构/训练步数。即使缓存满足速度，仍需必要种子和合法合格独立验证；若预算内未通过，报告NOT_ESTABLISHED及具体缺口，不因GPU尚空闲盲目训练。
最近资源快照（并非停止时最终结算）：{"GPU_wall_hours_including_live": 11.07713697756744, "home_free_GiB": 50.02982711791992, "new_retained_MiB": 311.70238971710205}。累计上限12 GPU-wall-hours、新留存512MiB、目标盘至少30GiB余量，重设Goal不清零预算。

## 10. 代码、记录与GitHub

[研究分支当前报告](https://github.com/LYQ1107/WWW/blob/research/gmt-jev-risk-association-20261014/docs/GMT_JEV/FINAL_RESEARCH_REPORT_ZH.md)。原始完整GTA checkpoint、视频、图像、预测压缩包和训练权重留服务器，仅推必要源码和紧凑记录；各自SHA已记录。
运行起点HEAD和源码SHA见 EXPERIMENT_REGISTRY.json，具体结果含source binding。发布是否成功以runtime/publication/LAST_VERIFIED.json的local/remote HEAD一致为准；报告生成HEAD不是未来发布提交号。
关键证据：BASELINE_PARITY.json、FULL_TRAIN_MATCHED_QUALITY_AUDIT.json、LEARNED_GATE_SUPERVISION_PRECHECK.json、LEARNED_GATE_SMOKE_TRAINING.json、LEARNED_GATE_RESULTS.json、LEARNED_FIRST_DIVERGENCE_NATIVE.json、SPEED_METRICS.json、GO_NO_GO.json。
