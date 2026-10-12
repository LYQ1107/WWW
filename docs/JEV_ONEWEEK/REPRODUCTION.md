# 本周实验复现与运行检查

分支 `jev/www-jev-oneweek-20261012`，工作树 `/home/liuyeqiang/WWW_jev_oneweek_20261012`，解释器 `/home/liuyeqiang/anaconda3/envs/GMT/bin/python`。运行根目录必须是该工作树。实际配置见 `configs/JEV_ONEWEEK/RUN_V1.json`；预注册全文、数据/权重路径与SHA分别见 `PREREGISTRATION.json`、`REUSED_EVIDENCE_REGISTRY.json`、`SUPERVISION_AUDIT.json`。旧阶段均只读。

执行状态：

```bash
/home/liuyeqiang/anaconda3/envs/GMT/bin/python reproduction_tools/jev_oneweek_controller.py --status
```

该状态同时显示每个RUNNING进程PID/独立日志、训练step、真实在线frame、数据留存及研究判断。逐任务原始日志位于 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1/controller_logs/`；队列日志的Traceback需要与registry及独立任务结果共同核查。GitHub发布失败不等于模型训练失败。此前发布超时已有修复和完整失败记录；同设置恢复会跳过完整TRAIN+native+MATLAB工件，不重训碰运气。

有限三种子正式队列：

```bash
/home/liuyeqiang/anaconda3/envs/GMT/bin/python reproduction_tools/jev_oneweek_queue.py --gpu 3 --arms jev threshold linear state_threshold --seeds 20261009 20261008 20261010
/home/liuyeqiang/anaconda3/envs/GMT/bin/python reproduction_tools/jev_oneweek_queue.py --gpu 4 --arms attention deepsets --seeds 20261009 20261008 20261010
/home/liuyeqiang/anaconda3/envs/GMT/bin/python reproduction_tools/jev_oneweek_queue.py --gpu 5 --arms mlp fixed_question --seeds 20261009 20261008 20261010
/home/liuyeqiang/anaconda3/envs/GMT/bin/python reproduction_tools/jev_oneweek_queue.py --gpu 6 --arms cross_only stability_only additive no_question_reader --seeds 20261009 20261008 20261010
```

这些是复现命令，不能对已经RUNNING的任务重复启动。每个模型fresh1000更新后继续同optimizer到累计4000，随后三个完整DEV视频及六相机pooled TrackEval/官方MATLAB；全部保留三个种子，固定最终checkpoint，无DEV调参。

规则与真实旧模型参考的队列使用 `jev_oneweek_control_queue.py`。`parent/cosine/fixed_threshold/fixed_fusion/extra_hungarian`、`XIII_fixed_reference/XV_V3_reference` 和 `base_single/base_repeat/base_rule/edge_single/edge_repeat/edge_rule` 是明确不同范围。XV V3只使用真实存在seed09；规则受对应冻结XIV初始化影响所以保留三种子。评分×求解seed09是诊断，不宣称原完整GMT传统cascade。

证据屏蔽 `no_cross_evidence/no_stability_evidence` 在公共旧编码器和新编码器之前应用，包括commitment_features；继承同一已训练JEV权重，属于冻结权重干预，不是独立重新训练。法定Global ID范围不改变，合法完整方案随被屏蔽评分重构，不能把此对照解释成只改变最终评分。

完整视频效率使用 `jev_oneweek_efficiency_queue.py` 等待GPU5自己的正式队列完成，随后独占同一物理V100。每种模型三次新构建/冷启动完整1200scene帧，预热8scene帧后重置，actual image IO→tracks，没有感知缓存命中、GT或轨迹日志。scene FPS是1200/秒，camera payload FPS是2400/秒。共享重复编码同权重另测；组件profile另跑完整视频，CUDA同步会改变成本，不能将其FPS当主效率值。参数读数使用真实TRAIN12图像首2帧，所有自定义核未完整覆盖的FLOPs如实PARTIAL。

同状态、标签认证、校准及索引诊断对应 `jev_oneweek_paired_native.py`、`jev_oneweek_budget_probe.py`、`jev_oneweek_trained_contract.py`、`jev_oneweek_question_separation.py`、`jev_oneweek_identity_matrix.py`。H64固定prefix32/0所有未来重新决定，与独立冷启动逐行N01/N10不同。精确预算仅按自然事件key哈希选择同一批事件、不读取GT/未来/有利分数；强制best非基线计划是诊断，不替代原策略运行。

发布：

```bash
/home/liuyeqiang/anaconda3/envs/GMT/bin/python reproduction_tools/jev_oneweek_publish.py --commit-message 'Record completed JEVTrack evidence milestone'
```

发布加互斥锁、更新主表/消融/参数/跨相机/Goal/十二问题报告，保留失败及原始SHA，然后普通push并核对远端HEAD。网络子进程组受有界timeout和三次重试控制；不强推。各科学工件仅从完成的结果提取，NOT_RUN/PARTIAL不填零。模型权重、图片、完整轨迹和详细episode仍保留服务器，没有大文件上传。
