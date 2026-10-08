# Phase IX native candidate values

**生产评分→全局 assignment→真实提交核心通过限定验收；整个因果数据到生产输入的桥接失败，不能开始模型训练。** 最终 Gate 见 PHASE9_CANDIDATE_SUBMIT_PARITY.json；早期 NATIVE_COMMIT_PARITY.json 保留它当时的完整视频实验范围。

配置默认 CANDIDATE_ENABLED=false。显式 opt-in 要求 MODEL.JEV.ENABLED=true 和 MODE=off，以固定原 MEMORY/REACT 规则及 trajectory RNG。入口是 VisionTrack 实际 sliding_inference_GMT → run_first_tracker_plus/run_global_tracker_plus → MATCH → bank/birth/MEMORY → 下一帧 live get_asso。单摄像头 sliding_inference 是另一个历史路径，opt-in 明确 fail fast，未声称集成。

评分、求解和提交分别由 jev_candidate_features.py、jev_candidate_policy.py、jev_candidate_assignment.py 与 GTRRCNN 原生路径承担。CandidateBatch 包含 raw score[D,C]、state64[D,64]、evidence12[D,C,12]、合法 mask、候选阈值、runtime references；模型只得到前三项数值输入/合法 mask，不得到身份整数。evidence12=GMT score、proposal margin、is_proposed、其他行竞争数、window observations、memory observations、track hits、view fraction、active flag、bank eligibility、gallery cosine、gallery available。当前 callsites 未提供 view fractions，该字段保留0；候选全部来自 active window，active flag=1。不能把占位字段说成已有丰富线上测量。未执行特征标准化训练。

model producer 接口为 (state64,evidence12,mask)→(values[D,C],NEW[D])；配置可加载明确提供的 TorchScript，没有新的训练 checkpoint。固定 gmt_values 用 raw score−原阈值、NEW=0，bidirectional 用行/列 softmax 均值、NEW=.5；这些只是已实现控制，不是已测性能。gmt_compat 单独保留 rectangular existing-only Hungarian 后按 candidate length 阈值拒绝，目标与 augmented values 不同，仅作 GMT OFF anchor。

所有 future learned/rule values 共享同一全矩阵 augmented Hungarian，每行一个 private NEW dummy，按原生 camera scope 限制重复身份，跨相机合法复用不被禁用。references 排序只确定等值解，不作为模型特征。显式 semantic NEW 跳过本行 stale-bank 恢复，再走真实 birth counter；compat unmatched 仍走旧 bank。existing 选择走原生 gallery/hits/MEMORY。没有研究脚本 monkey-patch 求解器；commit observer 仅收集实际容器。非有限 score 在进入 legacy scipy 前就受到 opt-in finite guard，旧关闭路径保持旧调用。

20项测试通过：空/全 mask/无检测/无 active、duplicate IDs、NaN/Inf、NEW shape/finite、4096候选、camera capacity、冲突全局选择、列排列/等值解、identity references 不进入模型、真实 GTR 预求解非有限防护和非 GMT 入口 fail fast。完整 video12=1200帧/2399次提交，GMT OFF 与 compat 全 Instances字段、ID计数、hits、gallery、bank、RNG哈希和 postprocessed outputs 相同。最新源码8帧回归：existing 替代改变13/15次提交，semantic NEW 改变15/15次，后续 live proposal score变化14次。都是零拟合参数的确定性 Torch producer，没有学习或性能主张。8帧低于原 min_track_len50，最终短段输出为空，不把其 SHA 相等当身份质量证据。

单个 early video16 前缀补首向量可使 evidence12 对齐；全数据221前缀中94个需要额外恢复 bank-reactivation 写入，适配器不成立。state64 全量桥接和完整生产 H32 干预等价未验证。该单前缀报告曾有过宽的 training-authorized 标志，最终修正为false，原文件/哈希另存保留，测得数值和原 FAIL 未改。停止 formal learning，不能从接口核心 PASS 推出 A/B 全部通过。

**WHAT DID WE LEARN?** 实际 ID 提交/后续 proposal 改变证明接口具有作用，但离线训练状态若缺 gallery 写入，shared solver 的正确性无法保证学习输入或反事实监督符合部署。
