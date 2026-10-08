"""Publish scoped Phase IX evidence and stop ineligible experiments honestly.

No tracker execution, model fitting, TEST or controller-heldout access occurs.
Earlier numeric support remains intact; final eligibility incorporates the
subsequent full-prefix representation failure.
"""
import collections
import json
from jev_phase9_common import *


def read(name):
    return json.loads((REPORTS / name).read_text())


def document(name, body):
    (ROOT / 'docs' / name).write_text(body.strip() + '\n')


def main():
    protect()
    support = read('SUPPLEMENTAL_CAPTURE_AUDIT.json')
    tests = read('CANDIDATE_INTERFACE_TESTS.json')
    production = read('NATIVE_COMMIT_PARITY.json')
    regression = read('CANDIDATE_PRODUCTION_REGRESSION.json')
    memory = read('MEMORY_REPRESENTATION_AUDIT.json')
    attribution = read('GALLERY_FAILURE_ATTRIBUTION.json')
    assert support['true_verified_corrections'] == {'train': 74, 'validation': 59, 'supplemental': 46}
    assert tests['status'] == 'PASS' and tests['tests'] == 20
    assert production['full_video_complete'] and production['tested_frames'] == 1200
    assert regression['status'] == 'PASS_BOUNDED_PRODUCTION_CONTRACT'
    assert memory['status'] == 'FAIL_PREFIX_REPRESENTATION_CONTRACT'
    assert memory['snapshots'] == 221 and len(memory['failures']) == 94
    assert attribution['formal_learning'] == 'BLOCKED_DEPLOYMENT_CONTRACT'
    protocol_sha = sha(REPORTS / 'PHASE9_SUPPLEMENTAL_CAPTURE_PROTOCOL.json')
    assert protocol_sha == '5d8504af27a351346e683aeeec1bb67678a71249b213a181bd6bb020d788a966'
    source = binding()
    final = OUT / 'finalization_v1'
    final.mkdir(exist_ok=True)

    # Keep the original one-prefix result as a separate immutable artifact.
    # Its old training-authorized flag was too broad; one passing prefix
    # cannot authorize fitting after a failed whole-data representation audit.
    original = final / 'ORIGINAL_FEATURE_BRIDGE_RESTORATION_PARITY.json'
    path = REPORTS / 'FEATURE_BRIDGE_RESTORATION_PARITY.json'
    if not original.exists():
        original.write_bytes(path.read_bytes())
    restoration = read('FEATURE_BRIDGE_RESTORATION_PARITY.json')
    restoration['formal_training_authorized_by_this_test'] = False
    restoration['final_qualification'] = {
        'status': 'SINGLE_PREFIX_PASS_ONLY_FULL_DATA_BRIDGE_FAILED',
        'original_report': {'path': str(original), 'sha256': sha(original)},
        'original_scoped_authorization_flag': True,
        'reason': '94 of 221 prefixes fail the birth-only adapter; retain the measured PASS but revoke its overbroad fitting flag',
        'measured_evidence_unchanged': True,
    }
    save(path, restoration)

    refs = ['CANDIDATE_INTERFACE_TESTS.json', 'NATIVE_COMMIT_PARITY.json',
            'EXISTING_CHOICE_PARITY.json', 'CANDIDATE_PRODUCTION_REGRESSION.json',
            'FEATURE_BRIDGE_PARITY.json', 'FEATURE_BRIDGE_RESTORATION_PARITY.json',
            'MEMORY_REPRESENTATION_AUDIT.json', 'GALLERY_FAILURE_ATTRIBUTION.json']
    joint = {
        'status': 'BLOCKED_DEPLOYMENT_CONTRACT', 'binding': source,
        'solver_and_producer_tests': {'status': 'PASS', 'tests': 20},
        'full_video_compatibility': {'status': 'PASS', 'video': 12, 'frames': 1200,
            'commits': production['outputs']['GMT_OFF']['commits'],
            'all_Instances_fields_hits_gallery_bank_birth_RNG': True,
            'postprocessed_outputs_equal': True},
        'bounded_latest_source_mutation': {'status': 'PASS', 'frames': 8,
            'existing_choice_changed_commits': regression['existing_values_deployed_changed_commits'],
            'semantic_NEW_changed_commits': regression['semantic_NEW_deployed_changed_commits'],
            'subsequent_live_score_changes': regression['next_frame_live_score_matrix_changes'],
            'producer': 'deterministic Torch modules with zero fitted parameters',
            'postprocessed_short_segment_performance': None},
        'full_causal_data_to_production_bridge': {'status': 'FAIL',
            'snapshots': 221, 'failed_birth_only_translation_prefixes': 94,
            'track_prefix_instances': memory['per_track_instances_checked'],
            'offset_histogram': memory['offset_histogram'],
            'state64_full_parity': 'NOT_VERIFIED',
            'native_counterfactual_H32_after_bank_events': 'NOT_VERIFIED'},
        'training_authorized': False, 'learned_closed_loop': 'NOT_RUN',
        'default_candidate_enabled': False, 'production_entry': 'sliding_inference_GMT',
        'non_GMT_entry': 'FAIL_FAST_WHEN_OPTED_IN',
        'protected_lifecycle': 'MODEL.JEV.ENABLED=true, MODE=off',
        'evidence': [{'path': str(REPORTS / n), 'sha256': sha(REPORTS / n)} for n in refs],
        'WHAT_DID_WE_LEARN': 'A working shared production assignment/commit core does not prove the causal replay gallery and training inputs are native-equivalent.',
    }
    save(REPORTS / 'PHASE9_CANDIDATE_SUBMIT_PARITY.json', joint)
    save(REPORTS / 'PHASE9_CANDIDATE_INTERFACE_TESTS.json', tests)

    qualification = {
        'status': 'BLOCKED_DEPLOYMENT_CONTRACT',
        'numeric_replay_support': 'PASS_SUPPLEMENTAL_DATA_SUPPORT',
        'training_authorized': False,
        'reason': 'Actual production gallery and stored causal replay memory differ; the birth-only adapter fails 94/221 prefixes and native H32 state equivalence after bank events is unverified',
        'evidence': ['PHASE9_CANDIDATE_SUBMIT_PARITY.json', 'MEMORY_REPRESENTATION_AUDIT.json', 'GALLERY_FAILURE_ATTRIBUTION.json'],
        'counts_labels_and_failed_outcomes_preserved': True,
        'no_claim_that_every_H32_label_is_wrong': True,
    }
    audit = read('CAUSAL_DATASET_AUDIT.json')
    audit['A_gate_scope'] = 'research replay paired-control, legal helper commit and numeric support only; full deployed state qualification is separately failed'
    audit['final_qualification'] = qualification
    save(REPORTS / 'CAUSAL_DATASET_AUDIT.json', audit)
    manifest = read('PHASE9_CAUSAL_DATASET_MANIFEST.json')
    manifest.setdefault('original_numeric_audit_sha256', manifest['audit_sha256'])
    manifest['audit_sha256'] = sha(REPORTS / 'CAUSAL_DATASET_AUDIT.json')
    manifest['final_qualification'] = qualification
    save(REPORTS / 'PHASE9_CAUSAL_DATASET_MANIFEST.json', manifest)

    blocked = {'status': 'NOT_RUN', 'blocking_gate': 'G_NATIVE',
        'blocking_result': 'BLOCKED_DEPLOYMENT_CONTRACT',
        'reason': qualification['reason'], 'binding': source,
        'trained_models': 0, 'trained_checkpoints': 0, 'metrics': None,
        'Full24': False, 'official_TEST': False, 'heldout_sealed': True,
        'WHAT_DID_WE_LEARN': 'No model, supervision or generalization conclusion is valid before the full production input/state bridge passes.'}
    specs = {
        'TINY_OVERFIT.json': ('C', ['CandidateMLP', 'DeepSets', 'CandidateJEV']),
        'BASELINE_COMPARISON.json': ('D0', ['GMT_OFF', 'fixed_rule', 'learned_threshold', 'bidirectional', 'B2_MATCH_ONLY']),
        'ARCHITECTURE_ABLATION.json': ('D', ['CandidateMLP', 'DeepSets', 'CandidateJEV', 'CAMEL_inspired_context']),
        'HELDOUT_RESULTS.json': ('E', [20, 21, 22]),
        'PHASE9_TRAINING_ABLATION.json': ('D', [5, 20, 50, 100]),
        'PHASE9_DATA_SCALING.json': ('D', [.25, .5, .75, 1.]),
        'PHASE9_CAPACITY_ABLATION.json': ('D', [8000, 34000, 128000, 500000]),
        'PHASE9_SUPERVISION_ABLATION.json': ('D', ['correctness', 'H32_utility', 'joint', 'ranking_advantage']),
        'PHASE9_NORMALIZATION_ABLATION.json': ('D', ['raw', 'train_stat', 'LayerNorm']),
    }
    for name, (stage, plan) in specs.items():
        obj = dict(blocked, phase=stage, planned_comparisons=plan)
        if name == 'TINY_OVERFIT.json':
            obj.update(optimization_failure_observed=False,
                known_NEW_correctness_positive_labels=0,
                NEW_label_limitation='Secondary identifiability issue: all current NEW labels are false or unknown; no positive labels invented. This is not a measured overfit failure.')
        if name == 'HELDOUT_RESULTS.json':
            obj.update(controller_heldout_predictions_opened=False,
                foundation_training_exposure_caveat='Controller-heldout split does not establish that GMT foundation pretraining excluded these TRAIN videos; whole TRAIN annotation metadata are used by the existing loader, but no heldout controller predictions/metrics or model selection were read.')
        save(REPORTS / name, obj)
    save(REPORTS / 'PHASE9_TINY_OVERFIT.json', read('TINY_OVERFIT.json'))

    answers = {
        '1_old_JEV_vs_rules': 'Phase VII same-solver controls did not establish a state-only JEV advantage. Candidate-specific evidence and corrective support were missing, while rules had the same native re-solve opportunity. This motivates a hypothesis; it does not isolate architecture as the cause.',
        '2_supplemental_support': 'Numeric replay support improved: train74, supplemental46/48, combined validation59/61, contributing train17/validation9 conflict groups. The production-state qualification remains blocked; repeated rows and weight ESS are not independent replications.',
        '3_MLP_sufficient': 'NOT_TESTED: no fitting or eligible fair closed-loop comparison.',
        '4_context_explains_gain': 'NOT_TESTED: external source review is not measured DeepSets/CAMEL-inspired performance.',
        '5_question_action_contribution': 'NOT_TESTED: MATCH-only fixed WHO question cannot identify a question-conditioned contribution without equal-context ablations.',
        '6_utility_vs_correctness': 'NOT_TESTED: distinct labels and executed replay branches are preserved; deployed state equivalence and equal-network supervision ablations are still required.',
        '7_defensible_contribution': 'Bounded conflict-aware causal sampling/auditing, an opt-in shared native candidate-values core, and an identified replay/native state failure are reproducible engineering/research evidence. JEV architecture and WWW publishability are not established.',
    }
    final_gate = {'status': 'STOP_BLOCKED_DEPLOYMENT_CONTRACT', 'binding': source,
        'conditional_audit_task': 'COMPLETE_AT_FAILED_GATE',
        'research_hypothesis': 'UNRESOLVED', 'JEV_architecture_rejected': False,
        'gates': {
            'G_DATA': {'status': 'PASS_NUMERIC_REPLAY_SUPPORT_ONLY', 'full_native_qualification': 'NOT_VERIFIED', 'training_authorized': False},
            'G_NATIVE': {'status': 'BLOCKED_DEPLOYMENT_CONTRACT', 'production_values_core': 'PASS_SCOPED', 'causal_data_state_bridge': 'FAIL'},
            'G_TINY': {'status': 'NOT_RUN'}, 'G_ARCH': {'status': 'NOT_RUN'},
            'G_HELDOUT': {'status': 'NOT_RUN_SEALED'}, 'G_LIFECYCLE': {'status': 'NOT_RUN_BLOCKED_UNIFIED_LIFECYCLE'}},
        'seven_answers': answers, 'models_trained': 0,
        'Full24_authorized': False, 'official_TEST_authorized': False,
        'heldout': [20, 21, 22], 'heldout_sealed': True,
        'B2_sha256': B2_SHA, 'phase8_original_gate': 'FAIL_UNCHANGED',
        'next_minimal_hypothesis': attribution['next_minimal_hypothesis'],
        'next_hypothesis_executed': False,
        'WHAT_DID_WE_LEARN': 'Sampling can remove the observed numeric shortage without removing a native state-contract failure. Stop before optimization; no architecture verdict can be drawn from untrained models.'}
    save(REPORTS / 'FINAL_GO_NO_GO.json', final_gate)

    document('PHASE9_CAUSAL_DATASET_REPORT.md', f'''
# Phase IX causal dataset audit

**结论：补充样本的回放数量门槛已达成，完整生产状态资格未通过；正式训练未执行。**

基础 Phase VIII `{BASE}` 保持原 TRAIN 122211 / validation 38887 MATCH events、可行候选 2278/118、原因果事件 82/13、纠错74/13和 validation FAIL。Phase IX 使用独立 worktree/runtime，未追认旧 Gate。

冻结协议 SHA256 `{protocol_sha}`。TRAIN12/13/14/16，validation17/18/19，controller-heldout20/21/22 sealed。补充48来自既有118个可行索引，先排除原13，再按完整候选引用、历史锚定目标、同一状态和 H32 时间依赖形成传递冲突组。种子20261009，视频配额17/14/17、组内无放回抽样；未来 utility 之前已提交冻结方案，失败与 unknown 全保留。

| 回放支持 | 审计事件 | 核验纠错 | 贡献视频 | 核验冲突组 |
|---|---:|---:|---|---:|
| 原 train | 82 | 74 | 12/13/14/16 | 17 |
| 原 validation | 13 | 13 | 17/18/19 | 9 |
| 新 supplemental | 48 | 46 | 17/18/19 | 8 |
| combined validation | 61 | 59 | 17/18/19 | 9 |

验证按视频核验24/14/21（原+补充）。新48中2个 video18 事件未满足立即正确提交等完整纠错条件，仍在 audit 中；不能因为全部 CANDIDATE_0 H32 为正就把这两行算成合格纠错。原13、补充48、合计61的组权重 Kish ESS 分别11.5714/35.3135/43.6574；这些数不是独立重复次数，真正组数为9/8/9，且仅三个验证视频。

预注册同时采集 natural56（train32/val24）、hard negative13（10/3）、unknown14（8/6）；加补充48，共131条采样记录/131行、130个唯一状态快照。部分视频达不到每视频4个困难负例，使用全部实际可用事件，未人工创造样本。自然分布的采样概率保留，候选级普通指标/神经模型自然分布性能未计算，不能用纠错富集分布代替总体收益。

有界未来预算执行67个新事件、354个分支；原95保留，compact audit 共162个事件。CONTROL/KEEP_FACTUAL 使用相同完整回放状态和 RNG，各 CONTROL 的实际 ID 与完整事实流一致。干预包括候选、替代、共享求解 REASSOCIATE、semantic NEW；之后重新计算 live GMT proposal。H8/H16/H32、实际提交、行级立即正确性、全场效用及 birth penalty=0 敏感性分别记录。效用继承 Phase VIII 冻结权重：correct − wrong − 5 merge − .25 birth。未执行候选效用始终 null；依赖其他行的完整 assignment 效应不等同穷举 Q 函数。错改、负效用、tie、unknown 未删除；身份错误时长按 camera-frame 记录，未假定 FPS 转秒。

候选 correctness 来自永久前缀 anchor，整数 GT ID 不与 track ID 直接比较；多正确 alias/unknown 保留。在线候选来自实际 GMT，全候选集，无 GT Top-K 保留。GT 仅用于离线审计选择和标签。当前新捕获行已有1635条已知 existing-candidate correctness 标签，但 NEW 正确性没有正例（train44 unknown/6 false，val70 unknown/11 false），不能伪造 NEW 监督，也不能把未训练称为优化失败。

**后续完整桥接审计改变了训练资格。** 直接将 replay.memory 当生产 gallery，train16/[17,1] 的 evidence12 最大偏差1.0。恢复真实首观测后该单前缀误差降至2.38e−7，但221个全前缀中94个不满足仅补首向量的契约。共39536个身份前缀，hits−writes=1/2/3分别39296/234/6。生产 birth 初始化 gallery；回放 memory 不含该项；bank 恢复后生产还写入观测，回放先筛 final_existing 的 MEMORY 阶段会漏掉恢复行。video12/[839,1] 身份5恢复，在[851,1]生产 gallery1367条、回放1365条，证据见 GALLERY_FAILURE_ATTRIBUTION.json。

这证实输入/状态映射尚未统一，不证明所有已有 H32 数值都错。普通首项偏移的 bank eligibility 阈值已有补偿，也不能直接声称所有 bank 支持都不同。完整 native 干预后 bank/gallery/normalization/state64 等价尚未获证，因此数字 PASS 仅限研究回放支持，G_NATIVE 为 BLOCKED_DEPLOYMENT_CONTRACT。

原始快照、效果和轨迹保留 `{OUT}`；845个新 dataset artifacts（130 state、354 EFFECTS、354 TRACE、7 fork manifest）的路径/SHA见 PHASE9_CAUSAL_DATASET_MANIFEST.json。原状态来源及哈希还见 MEMORY_REPRESENTATION_AUDIT.json。Git仅收 compact 标签、指标、清单、代码与文档。早期脚本 provenance/time metadata 错误运行亦保留，未作为科学失败样本删除。

**WHAT DID WE LEARN?** 事前分组补采能解决观察到的数量不足，但回放自洽和实际 production ID 一致都不能替代完整状态表示与线上输入的等价证明。
''')

    document('PHASE9_NATIVE_CANDIDATE_INTERFACE.md', '''
# Phase IX native candidate values

**生产评分→全局 assignment→真实提交核心通过限定验收；整个因果数据到生产输入的桥接失败，不能开始模型训练。** 最终 Gate 见 PHASE9_CANDIDATE_SUBMIT_PARITY.json；早期 NATIVE_COMMIT_PARITY.json 保留它当时的完整视频实验范围。

配置默认 CANDIDATE_ENABLED=false。显式 opt-in 要求 MODEL.JEV.ENABLED=true 和 MODE=off，以固定原 MEMORY/REACT 规则及 trajectory RNG。入口是 VisionTrack 实际 sliding_inference_GMT → run_first_tracker_plus/run_global_tracker_plus → MATCH → bank/birth/MEMORY → 下一帧 live get_asso。单摄像头 sliding_inference 是另一个历史路径，opt-in 明确 fail fast，未声称集成。

评分、求解和提交分别由 jev_candidate_features.py、jev_candidate_policy.py、jev_candidate_assignment.py 与 GTRRCNN 原生路径承担。CandidateBatch 包含 raw score[D,C]、state64[D,64]、evidence12[D,C,12]、合法 mask、候选阈值、runtime references；模型只得到前三项数值输入/合法 mask，不得到身份整数。evidence12=GMT score、proposal margin、is_proposed、其他行竞争数、window observations、memory observations、track hits、view fraction、active flag、bank eligibility、gallery cosine、gallery available。当前 callsites 未提供 view fractions，该字段保留0；候选全部来自 active window，active flag=1。不能把占位字段说成已有丰富线上测量。未执行特征标准化训练。

model producer 接口为 (state64,evidence12,mask)→(values[D,C],NEW[D])；配置可加载明确提供的 TorchScript，没有新的训练 checkpoint。固定 gmt_values 用 raw score−原阈值、NEW=0，bidirectional 用行/列 softmax 均值、NEW=.5；这些只是已实现控制，不是已测性能。gmt_compat 单独保留 rectangular existing-only Hungarian 后按 candidate length 阈值拒绝，目标与 augmented values 不同，仅作 GMT OFF anchor。

所有 future learned/rule values 共享同一全矩阵 augmented Hungarian，每行一个 private NEW dummy，按原生 camera scope 限制重复身份，跨相机合法复用不被禁用。references 排序只确定等值解，不作为模型特征。显式 semantic NEW 跳过本行 stale-bank 恢复，再走真实 birth counter；compat unmatched 仍走旧 bank。existing 选择走原生 gallery/hits/MEMORY。没有研究脚本 monkey-patch 求解器；commit observer 仅收集实际容器。非有限 score 在进入 legacy scipy 前就受到 opt-in finite guard，旧关闭路径保持旧调用。

20项测试通过：空/全 mask/无检测/无 active、duplicate IDs、NaN/Inf、NEW shape/finite、4096候选、camera capacity、冲突全局选择、列排列/等值解、identity references 不进入模型、真实 GTR 预求解非有限防护和非 GMT 入口 fail fast。完整 video12=1200帧/2399次提交，GMT OFF 与 compat 全 Instances字段、ID计数、hits、gallery、bank、RNG哈希和 postprocessed outputs 相同。最新源码8帧回归：existing 替代改变13/15次提交，semantic NEW 改变15/15次，后续 live proposal score变化14次。都是零拟合参数的确定性 Torch producer，没有学习或性能主张。8帧低于原 min_track_len50，最终短段输出为空，不把其 SHA 相等当身份质量证据。

单个 early video16 前缀补首向量可使 evidence12 对齐；全数据221前缀中94个需要额外恢复 bank-reactivation 写入，适配器不成立。state64 全量桥接和完整生产 H32 干预等价未验证。该单前缀报告曾有过宽的 training-authorized 标志，最终修正为false，原文件/哈希另存保留，测得数值和原 FAIL 未改。停止 formal learning，不能从接口核心 PASS 推出 A/B 全部通过。

**WHAT DID WE LEARN?** 实际 ID 提交/后续 proposal 改变证明接口具有作用，但离线训练状态若缺 gallery 写入，shared solver 的正确性无法保证学习输入或反事实监督符合部署。
''')

    document('PHASE9_ARCHITECTURE_DESIGN.md', '''
# Phase IX architecture hypotheses — NOT IMPLEMENTED / NOT TRAINED

所有神经架构保持待检验设计，不能把生产 Torch producer 接口或开源代码审计称作 Candidate JEV 训练成功。G_NATIVE bridge FAIL，C–F 均 NOT_RUN。

共同输入是同一在线 state64、候选 evidence12、全合法 mask；references 只负责提交。候选共享编码器逐条输出价值，同时从 detection context 输出 semantic NEW。禁止绝对 ID embedding、候选 GT 保留、未来片段 attention。每个模型均调用同一 augmented Hungarian、同一 native commit，MEMORY/REACT 冻结。正确性估计与效用估计是不同头/监督，未执行 utility 保持 unknown。

| 假设模型 | 拟议区别 | 需要排除的混淆 |
|---|---|---|
| Candidate MLP | 相同逐候选共享 MLP + 公共竞争字段、独立 NEW head | 不弱化输入或让 JEV 独占上下文 |
| DeepSets | 同候选编码器、masked pooling、集合上下文广播再评分 | 与 MLP 同特征/监督/预算，排列等变 |
| CAMEL-inspired context | 前缀 token 和候选交互；轻量 masked context，保持在线约束 | 不照搬外部困难采样并将其当因果标签，不用未来 |
| Candidate JEV | Question/Action-conditioned candidate value、correctness/utility heads、竞争上下文 | 与无 Q/A、同上下文普通头消融；WHO 固定不能独立证明 question 有用 |

已冻结训练计划：seeds20261008/09/10，约34K参数参照，epochs5/20/50/100，独立组数据25/50/75/100%，raw/train-stat/LayerNorm，LR=.001、weight decay=.0001、batch32、gradient clip5。用户要求8K/34K/128K/500K容量曲线视最初 fair/tiny 结果有条件实施，不无条件笛卡尔积扩大训练。实际参数、MACs、forward latency均null，因为网络未实例化/训练/评测，不能把计划容量当测量值。

监督消融在同一网络上比较 correctness、实际 H32 utility、joint、frozen ranking/advantage；普通 MLP/DeepSets必须同样享有因果监督。先验证tiny可以拟合可靠标签、合法概率/NEW/finite gradients和训练线上 normalization一致，再做正式公平比较。NEW当前没有可靠正正确性标签，需要单独完成资格，不能伪造正例。全候选优先；TRAIN-only recall@8/16/32/64/full后再讨论Top-K，当前不截断也不虚构recall。

评价应分别报告 natural/hard-corrective/hard-negative/unknown，group-weight、视频配对和统计不确定性；使用强普通 baseline、同 solver 和同重求解预算，之后在sealed20/21/22真实 mutated online检验HOTA/AssA/IDF1/IDSW/MOTA、camera-frame传播时长、误纠正、birth/merge、效用regret和延迟。当前这些模型比较均未执行，不存在JEV架构优劣结论。只有MATCH闭环合格后，才研究真实WRITE如何影响身份查询以及relative OLD/NEW reactivation，Unified当前禁止启动。

**WHAT DID WE LEARN?** Candidate conditioning、上下文交互和共享求解器都是普通可学习关联也具备的能力；JEV独立贡献必须经Q/A消融与公平监督比较，不能由名称或结构示意替代。
''')

    document('PHASE9_MODEL_COMPARISON.md', '''
# Phase IX model comparison

NOT_RUN / BLOCKED_DEPLOYMENT_CONTRACT。Candidate MLP、DeepSets、Candidate JEV、learned threshold及B2 MATCH-only的本阶段公平模型比较均未执行；参数/MACs、accuracy、Rank-1/MRR、regret、HOTA/AssA、训练曲线和置信区间全部null。

已完成GMT OFF→compat完整production parity及确定性值producer提交回归，属于接口检验。不能把它们当成模型训练、模型优势或完整closed-loop controller-heldout结果。

**WHAT DID WE LEARN?** 没有数据/部署资格就没有可解释的架构排名；阻塞证据见FINAL_GO_NO_GO.json与GALLERY_FAILURE_ATTRIBUTION.json。
''')

    document('PHASE9_FINAL_RESEARCH_REPORT.md', f'''
# Phase IX final research report

**本次有条件科研审计已完成：补采的研究回放数量门槛通过，生产 candidate-values 核心通过限定验收；完整数据→生产状态桥接失败，因此在 G_NATIVE 停止，C–F 全部 NOT_RUN。尚不能证明或证伪 JEV 架构的独立收益。**

代码基线 `{BASE}`，分支 `jev/www-jev-phase9-native-candidate-choice-20261008`，worktree `{ROOT}`，独立 runtime `{OUT}`。保护 Phase V–VIII 原 refs/报告与 B2 `{B2_SHA}`；旧 validation FAIL 永久保留。controller-heldout20/21/22 sealed，未启动 Full24，未读取官方TEST，历史checkpoint删除0，新训练模型/权重0。旧 loader 可读取整个TRAIN注释元数据；controller-heldout不保证foundation训练未见这些视频。

| Gate / 工作 | 真实结果 | 对后续的含义 |
|---|---|---|
| Phase0 | 11仓库固定commit、函数级审阅/许可证，43源码/许可证文件哈希 | 仅设计学习，无外部模型性能主张 |
| A frozen supplement | 48预选、46核验；combined59/61；train74；组17/9 | 数量及研究回放支持PASS，未获完整生产训练资格 |
| B shared production core | 20测试；video12完整1200帧/2399commit compat；8帧existing/NEW mutation | 评分→solver→commit及下一步proposal作用PASS，范围限定 |
| B full causal input/state bridge | 原evidence12不等价；单前缀补首向量PASS；94/221全前缀失败 | BLOCKED_DEPLOYMENT_CONTRACT |
| C tiny / D fair training+ablations | NOT_RUN，metrics=null | 无优化/架构胜负结论 |
| E independent online heldout / F lifecycle | NOT_RUN / SEALED | 无generalization或Unified增量主张 |

**具体失败原因。** production birth 初始化 gallery，回放 memory 从后续WRITE才开始。恢复真实首观测在video16/[17,1]可将evidence12误差从1.0降至2.38e−7；但全前缀审计发现有些bank恢复身份增加hits，却被回放在恢复前筛出的MEMORY写入阶段漏掉。video12/[839,1]身份5恢复；到[851,1]生产gallery1367、回放writes1365。39536个身份前缀中缺1/2/3项为39296/234/6，共94个快照不能用birth-only适配。这个失败限制训练/反事实部署等价，不证明每条历史H32标签错误；普通首项偏移的bank阈值本已有补偿。完整生产state64/归一化/干预后gallery/bank等价没有验收。

单前缀修复结果原有过宽的“training authorized”字段，最终改为false并保存原文件/哈希；数值PASS、最初FAIL、全量FAIL及所有负例均保留。没有为得到正向结果更换验证视频、改Gate、增大网络或重训。

**修改的代码及用途。** 新增三个production模块分开numeric evidence、producer及共享constrained assignment；GTRRCNN显式opt-in集成whole-camera augmented values求解、private NEW dummy、native existing提交、semantic NEW/bank边界和纯观察commit hook；配置默认关闭，候选开启要求JEV enabled/MODEoff，非GMT路径fail fast。原关闭路径保留原SciPy调用。新增冻结分组抽样、事实状态capture、有界H8/H16/H32干预、独立资格finalizer、真实production parity、gallery/state桥接审计和20测试。没有复制外部GPL代码或安装大型外部环境。

候选values比较共享augmented solver；GMT compat仅作为旧rectangular Hungarian→threshold语义锚点，其目标不同，未混作公平架构方法。确定性Torch producer不是学习模型；8帧低于min_track_len50，短段postprocessed输出为空，实际mutation证据是native commits及下一步raw scores。12字段中view_fraction目前为0保留位，不声称已实现该在线信息。

完整video12compatSHA `{production['outputs']['GMT_OFF']['full_output_SHA']}`。最新代码短回归existing改变{regression['existing_values_deployed_changed_commits']}/15次提交，NEW改变{regression['semantic_NEW_deployed_changed_commits']}/15次，后续live score改变{regression['next_frame_live_score_matrix_changes']}次。没有产生新的learned HOTA/AssA/IDF1/IDSW/MOTA或heldout延迟指标。

**七个研究问题。**

1. **旧JEV为什么被简单规则超过？** Phase VII没有建立state-only MATCH优于同solver规则的证据。缺候选相对信息、纠错支持不足、共享重求解机会是可检验解释；现有证据没有单独识别网络架构因果责任。
2. **补采解决样本不足了吗？** 数量层面是：新46/48、validation59/61、train74，17/9个贡献冲突组；但这不等于59个独立重复，也未解决完整production state资格。
3. **Candidate MLP足够吗？** 未检验，未训练，不能回答足够/不够。
4. **DeepSets/CAMEL context解释全部收益吗？** 未检验。外部实现说明context的合理机制，但不是VisionTrack测得收益。
5. **Question/Action有独立贡献吗？** 未检验；固定WHO的MATCH不能证明question conditioning，必须等数据合格后做同context去Q/A消融。
6. **长期utility比correctness更有价值吗？** 未检验。两类标签已分开、未执行效用为null，仍需native桥接后让所有普通模型享有同监督并做loss消融。
7. **最值得保留的论文创新是什么？** 当前可复核证据是有界冲突组因果采样/审计、共享生产候选值接口及状态表示失败定位。架构、长期决策收益和WWW论文贡献尚未建立，不能包装成已有胜利。

**最小下一步假设（本次未执行）。** 在固定预选事件上直接捕获真正production prefix并执行完整native H32 forks，或证明包含birth和reactivation WRITE的无损adapter；先复核同状态/RNG、state64/evidence12、gallery/bank/hits/计数、current assignment与mutated continuation。保持既有labels/结果/分割，只在新版本追加差异审计，再决定tiny。不能通过扩大模型、开放heldout或Unified绕过契约。

**存储和复核。** 原始证据约7.4 GiB仅保留服务器runtime，Git只存必要源码、compact JSON、配置、SHA和Markdown。dataset manifest含845个新raw artifacts，full-prefix audit另含221个状态哈希；production/feature reports保留各自原始trace路径和binding。早期provenance键/time-vector metadata脚本失败在原运行目录保留；纠正脚本在新版本继续，未覆盖失败。最终可运行verify_jev_phase9_delivery.py只读复核。reproduction_tools/finalize_jev_phase9_research.py记录报告生成逻辑；所有未获资格的计划指标为null。

代码/证据提交顺序：b9e2287外部审计→58be352事前协议→7aa78a8事实混合capture→2534eceproduction core→378d59a/5a487dd执行/严格提交资格→8bd2ca4/fe9e313真实existing继续→82a7391回放数量/完整compat结果→a880e01保留输入FAIL→d8e717f单前缀修复与全量审计→470e68d失败归因及入口防护→后续最终报告提交。审计绑定允许dirty标识并逐文件记录SHA，不能把有dirty的生成报告称作clean-run。

**WHAT DID WE LEARN?** 数据数量、solver正确性、生产提交作用和全量因果状态等价是四个不同条件。本次消除了前两类的部分缺口，也找到了第三类接口作用及第四类失败；在契约失败处停止，使后续实验仍然可解释。
''')
    save(final / 'REPORT_GENERATION_COMPLETE.json', {
        'status': 'COMPLETE_AT_FAILED_GATE', 'binding': source,
        'formal_training': 'NOT_RUN', 'final_gate_sha256': sha(REPORTS / 'FINAL_GO_NO_GO.json'),
        'new_metric_values_created_for_unexecuted_experiments': False,
    })
    print('FINAL_REPORTS_COMPLETE BLOCKED_DEPLOYMENT_CONTRACT', flush=True)


if __name__ == '__main__':
    main()
