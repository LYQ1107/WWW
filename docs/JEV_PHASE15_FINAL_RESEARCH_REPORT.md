# WWW / JEV Phase XV Final Research Report

**SCIENTIFIC_NO_GO。** 三版有界训练和真实原生诊断均已完成；合法 Commitment 在部分前缀存在纠错机会，但学习式控制尚未满足原生身份安全门。本次不是 WWW科研成功，也没有独立 Jev 架构优势结论。正式20k三种子和 matched4k 的条件不满足，未运行指标为null。

## 完成的证据链

P0 从0dc9607冻结 V–XIV，332个checkpoint原始hash以及83个受保护报告保留。P1 完成15个旧模型/视频实际原生重放，含 Multi三个种子和seed20261009的Fixed/Set，inputs/scores/IDs/最终状态完全一致。最坏种子824个CLEAR IDSW：pure fragment158、polluted history505、history corrective109、wrong existing merge30、false birth5、recovery1、ambiguous16；其中277个有合法纯净旧候选，只占33.62%，不能把所有切换都罚掉。

P2 冻结46个前缀、5个动作、2个后续策略，共460个声明分支；430实际执行、30个认证替代不可用。πMulti与πFixed分别有8/18、12/18个安全延续获益cluster，达到预冻结机制可行性门；cluster按video/GT/camera/64-frame block去重，不能当作总体统计独立样本。这是条件性实际因果证据，不是完整视频收益或架构优越性。KEEP目标及ALT兼容性由过去观测的GT离线认证，仅用于研究干预，不代表部署模型已识别该动作；任何动作仍受原生合法候选限制。TRAIN14原始GT重复导致严格CLEAR失败，另保留6个隔离的permissive诊断，排除严格门；原始失败不覆盖。DEFER多数窗口增加出生，强制KEEP也有混合增多窗口。

建立 GT-free Persistent CommitmentState，明确WHO vs具体native ID承诺，动态candidate-conditioned Q4，独立purity/safety/uncertainty，真实整个检测集合的Hungarian loss，原生 Gallery/Bank/ID/REACT/START_NEW不变。原 Multi四个完整TRAIN预测完全不变，512 payload恢复bitwise一致；另用第三版收集器验证512个学习式、带posterior状态payload恢复。Alias关闭；没有GT canonical renumbering。

## 有界修复与结果

详见 `JEV_PHASE15_TRAINING_AND_ABLATIONS.md` 和 `PILOT_RESULTS.json`。V1 缺必要纠错审计支持；V2 建立过去真实连续片段证据，必要纠错5/5但仍出现native污染；V3 使用V2自身完整TRAIN mutated-state历史，修复训练状态分布，并再次固定Tiny128/Pilot1500。全部版本及失败保留，没有挑最好checkpoint、丢最差种子、调DEV阈值或增加epochs。

V1相同权重、同prefix/RNG，仅屏蔽过去可靠性反馈的24个干预窗口没有任何实际ID变化；该证据只否定此通道在这些V1窗口的动作解释，不能覆盖V2/V3或未观察状态。原全局混合状态中大量查询缺少可认证纯净兼容候选；在 UNKNOWN上保持强延续可能延长身份混合，靠把UNKNOWN当负例无法可信解决。

具体反例：TRAIN12 frame3/camera1/row6前缀，V3在H32把IDSW减少4，却使新跨GT混合增加2、固定owner错误观测增加36、污染历史写入增加71。第一步仍延续可认证正确的ID9，后续策略自行产生的不安全提交才改变历史。它说明第一步延续正确及IDSW下降都不能保证闭环身份正确。这些数值是配对窗口差值；不同H及相邻窗口不当独立样本，精确输入与后续raw轨迹见NATIVE_UNSAFE_EVENTS.json的源文件引用。

出生口径：冻结的false_birth代理统计已出现GT再次START_NEW，可能包含必要恢复；false_split_birth另要求仍有可认证正确的合法旧候选。两者均保留，不把所有恢复出生自动判错。三版均存在实际新混合增加，故原生安全失败不只依赖该保守出生代理。

## 完整视频与官方指标

预先固定第三版LAST、seed20261009，在 DEV17/18/19完整当前图像→真实Stage1/VFCE→实际native Gallery/Bank状态运行。没有截断、短轨过滤、理想化teacher state或GT重编号。以下是诊断，不是合格正式三种子比较。

| 视频 | scene frames | HOTA | AssA | IDF1 | IDSW | MOTA | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|
| video17 | 1200 | 73.258 | 72.832 | 91.063 | 29 | 88.876 | 139 |
| video18 | 1029 | 73.341 | 71.339 | 87.654 | 41 | 92.047 | 62 |
| video19 | 1052 | 75.403 | 78.962 | 92.261 | 77 | 88.499 | 102 |

六个camera的 pooled raw TrackEval：

```json
{
  "AssA": 74.89070796515838,
  "DetA": 73.89697836976762,
  "Frag": 303.0,
  "HOTA": 74.18826995442981,
  "IDF1": 90.52104234028143,
  "IDSW": 147.0,
  "MOTA": 89.659419379163
}
```

原生官方MATLAB：**CVIDF1 90.285890, CVMA 86.781450**。使用未修改官方函数和已认证MEX；从每scene sequential身份计数求CVIDF1，从interleaved CLEAR计数求CVMA，不能平均video分数。历史 Phase XIV Multi seed09 HOTA73.476051/IDSW824/CVIDF187.195675；历史Fixed seed09 HOTA71.891351/IDSW206/CVIDF183.939320。历史三种子均值或不同阶段初始化不能替代本次公平A–F实验。

## 真正错误持续时间

在实际提交ID上固定第一次可观察GT为身份owner，包含真实最大摄像头bootstrap；后续混合仍累计错误。gap和视频结束右删失，不把缺帧当纠正；当前GT不能可靠匹配则unassessed。计量单位是实际错误观测帧，未验证真实采样帧率，不把它猜成秒。该offline observed-confusion定义与CLEAR/全局IDF1不同，完整episodes保存在服务器。

- video17, camera0, GT2, frame153–1135: 983个实际错误观测帧，right_censored=True。
- video17, camera1, GT2, frame546–998: 453个实际错误观测帧，right_censored=True。
- video17, camera1, GT2, frame335–543: 209个实际错误观测帧，right_censored=True。
- video18, camera1, GT10, frame654–887: 234个实际错误观测帧，right_censored=False。
- video18, camera1, GT2, frame367–466: 100个实际错误观测帧，right_censored=True。
- video18, camera0, GT7, frame890–983: 94个实际错误观测帧，right_censored=True。
- video19, camera0, GT7, frame217–263: 47个实际错误观测帧，right_censored=False。
- video19, camera1, GT6, frame553–583: 31个实际错误观测帧，right_censored=False。
- video19, camera0, GT10, frame842–865: 24个实际错误观测帧，right_censored=False。

## 真实速度与泛化

独立关闭GT/journals/evaluator，以真实图像输入重复3次DEV17前256scene frames，包括IO、detector、VFCE、history、policy、assignment、Gallery/Bank和posterior提交；此为共享GPU上的前缀运行，完整视频未审计的FPS为null，不用缓存FPS冒充。实际测量：

- Trial0: sceneFPS 2.713, total native Stage2 p95 68.407ms。
- Trial1: sceneFPS 2.618, total native Stage2 p95 68.612ms。
- Trial2: sceneFPS 2.621, total native Stage2 p95 68.938ms。

部署门还要求可靠的tracking质量；当前不具备部署资格。Stage1已在全部24个视频训练，DEV被历史研究使用。本次是关联控制器scene-disjoint训练和已使用DEV诊断，不能宣称绝对未见全系统泛化。20/21/22、官方TEST仍封存，Full24未启动，未把不合格外部WILDTRACK前缀冒充迁移证明。

## 科学结论与剩余阻塞

1. WHO正确候选集合与具体身份Commitment不同，存在合法可受益前缀；该机制可行性得到了原生配对支持。
2. 高认证延续准确率未足以解决全部查询的安全、错误后恢复及跨摄像头统一身份，三版均未达到完整native Pilot安全门。
3. 新增Q4和可靠性模块是否优于普通continuity或Fixed/Set同监督网络，本次没有资格开展正式公平对照，结论为未证实；不能声称等价，也不能声称Jev胜出。
4. 下一步必须先让无纯净兼容候选的状态具备可识别的纠错/恢复证据，区分必要出生与反复碎裂，检查视觉候选及原生生命周期约束。需新冻结预算和独立可认证纠错事件，再扩展架构或长训。对未知历史一律加负标签、靠总IDSW下降宣告成功或继续盲增epochs，都不构成当前阻塞的解决。

## 修改代码与复核

| 新增代码 | 具体变化 |
|---|---|
| `gtr/modeling/jev_phase15/commitment_state.py`、`native_commit_adapter.py` | GT-free过去承诺、版本化快照、原生提交后的posterior回写 |
| `commitment_question.py`、`commitment_option_reader.py`、`candidate_reliability.py`、`model.py` | 动态Q4、具体ID承诺、分离purity/safety/uncertainty；保留旧WHO初始化 |
| `joint_action_ranker.py`、`identity_fragment_controller.py` | 合法同容量分配接口；Alias关闭且不改写历史ID |
| `reproduction_tools/jev_phase15_commitment_labels.py`、`jev_phase15_losses.py`、`jev_phase15_train.py` | 过去纯净/局部不安全认证，UNKNOWN无伪负标签，完整检测集合联合损失与有界训练 |
| `jev_phase15_collect_pilot_onpolicy.py`、`jev_phase15_pilot_native.py`、`jev_phase15_all_query_audit.py` | 真实own-state采集、同prefix/RNG未来闭环、实际联合动作的风险覆盖 |
| `jev_phase15_evaluate.py`、`jev_phase15_matlab.py`、`jev_phase15_latency.py` | 完整图像原生评测、原始官方MATLAB指标、三次实际图像速度重启 |
| `jev_phase15_finalize.py`、`jev_phase15_delivery_check.py` | 条件阶段null结果、原始报告快照、SHA交付及只读复核 |

工程失败也保留：初版journal张量序列化；初版前缀活引用被后续Gallery/hits污染；原始GT重复导致严格CLEAR拒绝；旧联合损失删除UNKNOWN检测行；own-state observer在posterior写回前抓取最终状态；最终MATLAB汇总目录返回值解释错误及重试覆盖保护。修复均在新命名空间或明确版本下执行；原输出、源版本及日志未删除。汇总目录修复不重新训练，也不重新生成native预测。

在服务器工作树可运行只读复核：`python reproduction_tools/jev_phase15_delivery_check.py --all-phase15-refs`。检查20个必要JSON、6个文档及全部Phase XV本地SHA引用，复算官方指标聚合公式，核查三版Pilot均失败、正式梯度为0、三个完整视频和三次速度记录齐全；不会重跑实验或改写报告。机器之外的服务器产物只能按清单核对，不能凭GitHub小文件宣称已复现实验。


代码与紧凑JSON在独立 Phase XV branch；模型、数据、原始轨迹、失败和hash绑定文件留服务器。所有原始V–XIV研究资产仍保留，完整最终保护校验见FINAL_PRIOR_INTEGRITY.json。阶段条件及精确失败事件分别见FINAL_GO_NO_GO.json与NATIVE_UNSAFE_EVENTS.json。
