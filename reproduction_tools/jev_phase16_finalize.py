"""Honest complete delivery: actual diagnostics, conditional stages unrun/null."""
from jev_phase16_common import *
import collections


def main():
    protect();storage_guard()
    p1=read(REPORTS/'MULTI_PROTOTYPE_RESULTS.json');safety=read(REPORTS/'NATIVE_FUTURE_SAFETY_GO_NO_GO.json')
    labels=read(REPORTS/'LABEL_ELIGIBILITY.json');factorial=read(REPORTS/'MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json')
    pooled_path=OUT/'P8_pooled_memory_live/RESULT.json';pooled=read(pooled_path)
    runtime_path=OUT/'P8_latency_memory_only/RESULT.json';runtime=read(runtime_path)
    assert pooled['status']==runtime['status']=='COMPLETE'
    assert p1['selected_reader'] is None,'qualified reader requires conditional training, cannot finalize negative path'
    ck=read(XV/'training_full_payload_v3/F_full/seed20261009/pilot/RESULT.json')['checkpoint']
    sources=[ref(REPORTS/n) for n in ['MULTI_PROTOTYPE_RESULTS.json','NATIVE_FUTURE_SAFETY_GO_NO_GO.json','LABEL_ELIGIBILITY.json']]
    src=binding(checkpoints=[ck],inputs=sources,evaluator='frozen sequential evidence/safety/label gates',scope='conditional training NOT_RUN; frozen checkpoint actual diagnostic results retained')
    why=['No reader passes unchanged P1 gate','No qualified safe recovery reader available','Native safety and recovery-label qualification separately recorded']
    for name,planned in [('TINY_RESULTS',128),('PILOT_RESULTS',1500),('FORMAL_TRAINING_RESULTS',20000)]:
        save(REPORTS/(name+'.json'),dict(status='NOT_RUN_PREREQUISITE_FAILED',binding=src,reason=why,
            planned_updates=planned,executed_updates=None,seed=20261009 if name!='FORMAL_TRAINING_RESULTS' else None,
            planned_seeds=[20261009] if name!='FORMAL_TRAINING_RESULTS' else SEEDS,
            actual_checkpoint=None,training_metrics=None,online_metrics=None,safety_metrics=None,
            Phase_XV_frozen_checkpoint_is_not_new_training=True,failures_not_overwritten=True))
    save(REPORTS/'MATCHED4K_RESULTS.json',dict(status='NOT_RUN_PREREQUISITE_FAILED',binding=src,planned_updates=4000,
        planned_seeds=SEEDS,executed_updates=None,metrics=None,checkpoint=None,reason='No qualified formal/safe on-policy base'))
    save(REPORTS/'FAIR_BASELINE_RESULTS.json',dict(status='NOT_RUN_FORMAL_PREREQUISITE_FAILED',binding=src,
        Fixed_Question_same_evidence_20k=None,Set_Transformer_same_evidence_20k=None,
        formal_three_seed_new_JEV=None,dynamic_question_ablation=None,dynamic_option_ablation=None,
        frozen_native_factorial=ref(REPORTS/'MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json'),
        shared_XV_V3_checkpoint=True,independent_architecture_value_established=False,
        reason='Frozen same-network reader/admission attribution is measured; same-evidence learned Fixed/Set and new reader cannot train before evidence/safety/labels qualify'))
    live=[];propagation={}
    for video in DEV:
        p=OUT/'P8_memory_only_live'/f'video{video:02d}/RESULT.json';d=read(p)
        assert d['status']=='COMPLETE' and d['live_images'] and d['checkpoint']==ck
        live.append(dict(video=video,result=ref(p),metrics=d['strict_online_metrics'],
            legacy_single_GT_history_diagnostic=d['native_risk'],qualified_owner_propagation=d['error_propagation'],admission_counts=d['admission_counts'],
            raw_predictions=d['raw_predictions'],actual_mutated_state_online=True,full_image_cache_FPS_claim=False))
        propagation[str(video)]=read(d['error_propagation']['path'])
    historical_path=XV/'pilot_pooled_live_v3/RESULT.json';historical=read(historical_path)
    online=dict(status='COMPLETE_FROZEN_MEMORY_ONLY_DIAGNOSTIC',binding=binding(checkpoints=[ck],inputs=[ref(pooled_path)]+[r['result'] for r in live],
        evaluator='unmodified pooled raw TrackEval + actual image/native commits',scope='reused DEV17/18/19, seed20261009 frozen XV V3; no formal three-seed or unseen-system claim'),
        cases=live,pooled_raw_TrackEval=pooled['pooled_raw_TrackEval'],CVIDF1=pooled['CVIDF1'],CVMA=pooled['CVMA'],
        identity_error_propagation=propagation,preselected_rule='bounded_quarantine',trained_checkpoint=ck,
        exact_same_frozen_XV_reference=ref(historical_path),historical_XV_V3_pooled=historical['pooled_raw_TrackEval'],
        historical_XV_V3_CVIDF1=historical['CVIDF1'],historical_XV_V3_CVMA=historical['CVMA'],
        three_seed_new_model_worst_IDSW=None,three_seed_new_model_mean_IDSW=None,
        diagnostic_seed20261009_IDSW=pooled['pooled_raw_TrackEval']['IDSW'],
        native_risk_tag_scope='Legacy single-known-GT history tags supplementary only; primary owner propagation/split risk use >=3 and>=.8 coverage',
        full_cold_start_distinct_from_shared_legacy_prefix_admission=True,
        prior_failure_824_preserved=True,Stage1_has_seen_all24_TRAIN=True,independent_full_system_generalization=False,
        DEV_rule_or_threshold_selection=False,raw_unfiltered=True,official_TEST=False,Full24=False)
    save(REPORTS/'ONLINE_VALIDATION.json',online)
    save(REPORTS/'IDENTITY_ERROR_PROPAGATION.json',dict(status='COMPLETE',binding=online['binding'],
        full_video=propagation,paired_native_factorial=ref(REPORTS/'MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json'),
        immutable_qualified_owner=True,UNKNOWN_excluded_from_wrong_not_counted_correct=True,observed_gaps_censored=True,
        official_MOT_IDSW_distinct=True))
    latency_report=dict(runtime)
    latency_report['server_original_result']=ref(runtime_path)
    latency_report['component_sample_filter']='First16 invocations of each component excluded, as recorded actor source; not a uniform scene-frame timestamp filter'
    latency_report['total_Stage2_sample_filter']='First16 native payload invocations excluded after separate8-scene warmup'
    latency_report['null_Bank_promotion_scope']='No remaining samples after invocation filter; cannot infer no Bank calls or zero Bank cost'
    save(REPORTS/'LATENCY_RESULTS.json',latency_report)
    failures=[]
    for p in sorted(OUT.glob('P2_contract_engineering_attempt*.json')):failures.append(ref(p))
    save(REPORTS/'ENGINEERING_FAILURES_AND_REPAIRS.json',dict(status='RETAINED_AND_REPAIRED',binding=binding(scope='engineering provenance; no extra hypothesis or relaxed science gates'),
        P0_original_aggregation_path_repair=ref(REPORTS/'P0_REFERENCE_PATH_REPAIR.json'),
        P1_initial_slow_observer_restart=ref(REPORTS/'P1_ENGINEERING_RESTART.json'),
        P1_exact_old_partial_equivalence=ref(REPORTS/'P1_ENGINEERING_EQUIVALENCE.json'),
        P2_contract_attempts=failures,finalization_attempts=[ref(p) for p in sorted(OUT.glob('FINALIZATION_ENGINEERING_ATTEMPT*.json'))],final_native_parity=ref(REPORTS/'NATIVE_MEMORY_PARITY.json'),
        all_original_logs_retained=True))
    conclusion=dict(status='RESEARCH_COMPLETED_NO_GO',execution_status='COMPLETE_WITH_PREDECLARED_CONDITIONAL_NOT_RUN',
        binding=binding(checkpoints=[ck],inputs=[ref(REPORTS/'ONLINE_VALIDATION.json'),ref(REPORTS/'LATENCY_RESULTS.json')]+sources,
            evaluator='separate frozen tracking / independent architecture / deployment gates',scope='two bounded hypotheses tested; learning version unqualified'),
        P0='COMPLETE_GO_BOUNDED_FROZEN_RETRIEVAL_TRIAL',P1=p1['scientific_status'],P2='COMPLETE_GT_FREE_ADMISSION_AUDIT',
        P3=safety['status'],P4_new_evidence_conditioned_network='NOT_RUN_UNQUALIFIED',P5_labels=labels['status'],
        Tiny='NOT_RUN',Pilot='NOT_RUN',Formal_three_seed_20k='NOT_RUN',Matched4k='NOT_RUN',
        real_image_online='COMPLETE_MEMORY_ONLY_DIAGNOSTIC',official_MATLAB='COMPLETE_UNCHANGED',actual_latency='COMPLETE_THREE_RESTARTS',
        GO_TRACKING='FAIL',GO_JEV_INDEPENDENT_VALUE='NOT_ESTABLISHED',GO_DEPLOYMENT='PASS' if runtime['runtime_thresholds_pass'] else 'FAIL',
        tracking_blockers=['P1 no qualified reader','No qualified formal three-seed results','Native safety/label audit failed','Cold-start full-video diagnostic1494 IDSW and lower HOTA/AssA/CVIDF1; frozen rule seriously degrades tracking'],
        memory_safety_violations=safety['strict_worsened_prefix_H_checks'],memory_safe_benefit_independent_clusters=safety['independent_safe_benefit_clusters'],
        safe_recovery_label_clusters=labels['safe_recovery_independent_clusters'],unsafe_recovery_label_clusters=labels['unsafe_recovery_independent_clusters'],
        reserved_correction_clusters=labels['necessary_correction_reserved_clusters'],
        major_research_hypotheses_used=2,V3_not_started_due_to_failed_prerequisites=True,
        new_weight_checkpoint=None,prior_V_to_XV_results_and_weights_preserved=True,SEALED20_21_22='NO_VIDEO_CACHE_IMAGES_SELECTED_LABELS_OR_EVALUATION',official_TEST='UNREAD',Full24='NOT_RUN',
        shared_TRAIN_annotation_manifest='Existing train.json parsed/hashed by standard loaders; selected per-video observations and labels restricted to declared TRAIN/DEV',
        new_metrics_for_unrun_stages_are_null=True,scientific_blocker=None)
    blockers=dict(status='UNQUALIFIED_FOR_LEARNED_RECOVERY',binding=src,
        observed_information='Raw target-person content is widespread but Global ID ownership often contaminated; target content is not ownership certification',
        P1_discovery='Original support recall99.389% saturates the >=2pp gate; local readers do not prove additional recoverable support. Wrong activation reduction in shadow is not native safe benefit.',
        memory_discovery='Admission quarantine may change WHO and Q4 inputs and future IDs; decreased certified wrong counts can also reflect UNKNOWN coverage changes. Full paired coverage and unsafe branches retained.',
        unrecoverable_cases=['No observed target history','Person content only inside IDs with unknown or mixed ownership','Stale ID not legally offered by actual Bank','No distinguishing available history/candidate evidence'],
        next_evidence_needed=['Past-only owner-discriminating visual evidence and calibrated confusion margins','Verified cross-view temporal/spatial constraints','Independent reserved necessary-correction clusters with safe native outcomes'],
        Stage1_ReID_change='Not trained here. Consider only if same-camera/cross-camera positive-versus-foreign evidence remains inseparable in a separately frozen audit.',
        new_dataset='Needed for unseen-system claims because Stage1 has prior24-TRAIN exposure; this reused DEV cannot establish that claim.',
        JEV_direction='Continue only after identifiable safe correction supervision and same-evidence ordinary controller comparison; current architecture value not established.',
        ordinary_association='A simpler calibrated association/admission baseline remains scientifically justified; no claim that dynamic JEV is superior.',
        no_unsupported_third_training_version=True)
    save(REPORTS/'SCIENTIFIC_BLOCKERS.json',blockers)
    conclusion['scientific_blocker']=ref(REPORTS/'SCIENTIFIC_BLOCKERS.json')
    save(REPORTS/'FINAL_GO_NO_GO.json',conclusion)
    required=['FINAL_GOAL','PREREGISTRATION','EVIDENCE_AVAILABILITY_AUDIT','RAW_GALLERY_RECOVERABILITY','CANDIDATE_FAILURE_TAXONOMY','MULTI_PROTOTYPE_RESULTS','SAFE_MEMORY_ADMISSION','NATIVE_MEMORY_PARITY','MATCH_VS_MEMORY_CAUSAL_FACTORIAL','LABEL_ELIGIBILITY','TINY_RESULTS','PILOT_RESULTS','FORMAL_TRAINING_RESULTS','FAIR_BASELINE_RESULTS','ONLINE_VALIDATION','OFFICIAL_MATLAB_RESULTS','LATENCY_RESULTS','FINAL_GO_NO_GO']
    for name in required:assert (REPORTS/(name+'.json')).is_file(),name
    trials=runtime['trials'];metrics=pooled['pooled_raw_TrackEval'];s=p1['strict_TRAIN']
    sums=factorial['strict_window_sums'];factorial_rows='\n'.join(
        f"|{a}|{sums[a+'_H32']['future_CLEAR_IDSW']}|{sums[a+'_H32']['new_cross_GT_gallery_mix']}|{sums[a+'_H32']['wrong_owner_observations']}|{sums[a+'_H32']['UNKNOWN_owner_observations']}|" for a in 'ABCD')
    report=f'''# WWW / JEV Phase XVI 最终研究报告

执行完成；科研结论 **NO_GO**。完成 P0/P1/P2/P3、真实图像完整 DEV、官方 MATLAB 和实际延迟。没有合格依据启动新读取器、Tiny/Pilot、三种子20k或 matched4k；未运行指标为 null，既有 XV 失败与权重保持原样。

冻结起点 `{BASE}`。分支 `{BRANCH}`。V1 是冻结证据读取，V2 是冻结权重安全写入；V3 学习版因前置资格失败未启动。数据为 TRAIN12/13/14/16 与已反复使用的 DEV17/18/19。未使用20/21/22的视频缓存、图像、所选决策标签或评测；现有共享TRAIN标注train.json按标准loader解析并哈希，逐视频观测和标签提取限定在上述TRAIN/DEV。official TEST 未读取，Full24未执行。Stage1 已接触全部24个 TRAIN，不能声称全系统未见泛化。

## 证据与读取

P0 实际重放19个完整轨迹和72个 H32 分支。严格 TRAIN 的85793/85852条已知查询存在历史人物内容，但这不保证对应 Global ID 的纯净 owner。原生 REACT 仅在 MATCH DEFER 和真实合法 Bank 状态下调用前三槽最大余弦与0.75门槛；不是学习式 Q5，也不能将 DEFER 当安全拒绝。

P1 七个视频、五种四槽读取；冻结同一个 XV V3 seed20261009 LAST1500 权重，零训练更新，原始 logits、所有实际 ID 和最终原始记忆精确不变。A支持召回99.389%，剩余最大空间0.611pp，低于冻结+2pp门槛。这是协议与基线饱和的限制，不能降低门槛追认成功。C增益约0.145pp、D/E约0.076pp，全部不合格。局部原型改善部分影子错激活不能证明原生安全收益；A局部 token 纯净性未测量，报告null，不以初始化0作对比。

## 安全写入与因果归因

五个 GT-free 规则在相同实际 raw commits 上做影子比较。bounded_quarantine 在结果前选定；Gallery、hits、Bank、ID输出按各组真实原生动作更新并保留，原始观测不删除。独立匹配视图控制 WHO历史与 Q4前驱，CONFIRMED 只表示规则一致性。

24个原始前缀×四组=96条真实变异未来，H8/H16/H32全部保留，八个预定前缀额外H64。严格前缀18个，TRAIN14六个因重复GT只作诊断。A/B/C/D起始前缀、RNG、首次原始输入和原始分数相同；A全部H32 ID与XV原分支精确相等。C/D的E读取只干预一次目标决策，是失败读取器负对照，不是部署扩展。后续状态从未同步。

预选记忆规则门槛：**{safety['status']}**。严格有害prefix/H检查 {safety['strict_worsened_prefix_H_checks']} 个，安全收益独立cluster {safety['independent_safe_benefit_clusters']} 个（要求≥8）。所有有害路径、身份错误 episode 和交互效应保留在 `MATCH_VS_MEMORY_CAUSAL_FACTORIAL.json`。窗口求和存在重叠，不能当全视频或独立样本统计。

下面为18个严格前缀的H32描述性窗口求和，**窗口重叠，不是独立事件数或完整视频指标**。

|组|Future CLEAR IDSW|新增混合|错误owner观测|UNKNOWN owner观测|
|---|---:|---:|---:|---:|
{factorial_rows}

E的首次决策干预在这组前缀仅改变一条严格目标提交；其IDSW小幅变化没有产生合格的必要身份纠错标签。记忆组的混合和IDSW恶化，说明隔离匹配证据不能自动保证提交身份正确。最早反例H32短期改善仍在H64增加混合3次和false_split_birth1次，按预定规则否决。没有据此追改参数或挑选序列。

认证owner为过去首次单GT、至少三条、已知覆盖≥80%，建立后不因混合而取消。错误转UNKNOWN不算纠错。传播只统计实际检测到的错误帧，观测断点与horizon/video end删失；额外按每个GT/camera比较最长错误episode，避免总量掩盖个别人恶化。

标签资格：安全恢复 {labels['safe_recovery_independent_clusters']} cluster，不安全恢复 {labels['unsafe_recovery_independent_clusters']}，reserved必要纠错 {labels['necessary_correction_reserved_clusters']}，结论 **{labels['status']}**。不制造缺失恢复动作标签，不启动第三个学习版本。

这些恢复标签计数专指24个冻结前缀的目标首次实际动作与安全后果。大量原有正常关联监督和记忆规则的有害后果分别保留；后者不能冒充新JEV恢复动作标签。这是本轮有界资格审计，不是对所有可能TRAIN动作作穷尽搜索。

## 完整实际图像与官方指标

下面仅是预选安全记忆规则的冻结单种子诊断，使用实际图像→Stage1 VFCE→JEV→native commit→updated memory。完整原始预测不筛短轨、不改GT、不重编号。

|指标|XV冻结参考|Phase XVI安全记忆诊断|
|---|---:|---:|
|HOTA|{historical['pooled_raw_TrackEval']['HOTA']:.6f}|{metrics['HOTA']:.6f}|
|AssA|{historical['pooled_raw_TrackEval']['AssA']:.6f}|{metrics['AssA']:.6f}|
|IDF1|{historical['pooled_raw_TrackEval']['IDF1']:.6f}|{metrics['IDF1']:.6f}|
|IDSW|{historical['pooled_raw_TrackEval']['IDSW']}|{metrics['IDSW']}|
|MOTA|{historical['pooled_raw_TrackEval']['MOTA']:.6f}|{metrics['MOTA']:.6f}|
|Frag|{historical['pooled_raw_TrackEval']['Frag']}|{metrics['Frag']}|
|CVIDF1|{historical['CVIDF1']:.6f}|{pooled['CVIDF1']:.6f}|
|CVMA|{historical['CVMA']:.6f}|{pooled['CVMA']:.6f}|

各视频、认证覆盖、wrong-owner持续帧数、混合/出生、raw predictions SHA 在 ONLINE_VALIDATION.json 中。CVIDF1/CVMA 使用未修改官方 MATLAB、求和场景计数；完整toolkit哈希与历史一致。当前仅seed20261009，三种子新模型最差/均值均null。历史原关联最差824、XV V3的147完整保留，不能把单种子诊断当三种子恢复证明。

完整冷启动结果明确否决该规则：三个视频IDSW为320/431/743，合并1494，远高于XV参考147；HOTA、AssA和官方跨相机指标均严重下降。前缀实验共同保留旧历史，完整视频从冷启动使用新接纳规则，二者范围不同。这提示冷启动的隔离/fallback与改变后的WHO/Q4输入需要重新取得可靠资格；当前不能继续训练或部署该规则。主认证owner/false_split统计使用≥3与≥80%门槛，旧single-GT标签只作为明确标注的补充诊断。

## 实际运行成本

真实 DEV17前256双摄像头场景、三次重新启动、八帧warmup；无GT/evaluator/journal。场景FPS为 {', '.join(f"{t['full_two_camera_scene_FPS']:.3f}" for t in trials)}，Stage2 p95为 {', '.join(f"{t['total_native_Stage2_ms']['p95']:.3f}" for t in trials)} ms。共享GPU情况、同步开销和组件计时全部披露。完整视频无observer FPS未测量，null。附加序列化状态每ID最多六条外观向量，派生归一化缓存至多四条，新增持久外观向量总上界十条（≤16）；原始 Gallery和状态元数据仍随历史增长，不宣称总内存常数。

组件统计按实际actor代码排除每个组件前16次调用，并非统一按scene时间排除；总Stage2排除前16次native payload调用。Bank promotion在该过滤后没有剩余计时样本，null，不表示没有Bank调用或零开销。完整FPS包含真实图像读取和全部native路径，部署门槛没有通过。

## 最终资格与限制

GO_TRACKING **FAIL**；GO_JEV_INDEPENDENT_VALUE **NOT_ESTABLISHED**；GO_DEPLOYMENT **{conclusion['GO_DEPLOYMENT']}**。真实图像诊断的单指标改善不能覆盖原生安全、标签或三种子资格失败，也不能证明动态Question/Option优于普通Fixed/Set。同证据同监督正式比较未获资格，指标null。

核心阻碍是人物历史内容、可区分证据与合法Global ID owner之间的差距。没有历史、只有混合/UNKNOWN owner、真实Bank未提供、视觉证据缺乏区分力的情况无法由增加训练步数解决。下一步应先获得过去可验证的owner区分证据与足够独立reserved纠错事件；若正负外观仍不可分，再独立审计Stage1 ReID。更简单的校准关联/记忆规则应保留为对照，不为坚持JEV包装架构负结果。

## 可复核交付

工程失败和修复保留：P0原始reference路径异常仅CPU修复聚合；P1慢observer保留原始partial并与新轨迹逐条精确核对；P2两次接口合同失败日志保留并修复。没有改变科学门槛或消除原始失败。

代码位于 `gtr/modeling/jev_phase16/` 与 `reproduction_tools/jev_phase16_*`；报告在 `reports/JEV_PHASE16/`，原始大轨迹/日志/权重仅在服务器。各实际阶段绑定actor Git SHA、source/dataset/config/checkpoint SHA、seed、native前缀/RNG和evaluator环境。FINAL_GO_NO_GO.json 区分执行完成与科研失败，SCIENTIFIC_BLOCKERS.json列出后续证据需求。
'''
    (ROOT/'docs/JEV_PHASE16_FINAL_RESEARCH_REPORT.md').write_text(report)
    goal=read(REPORTS/'FINAL_GOAL.json');goal['status']='COMPLETE_EXECUTION_SCIENTIFIC_NO_GO';goal['execution_status']='COMPLETE_WITH_CONDITIONAL_NOT_RUN';goal['scientific_status']='NO_GO';goal['final_result']=ref(REPORTS/'FINAL_GO_NO_GO.json');save(REPORTS/'FINAL_GOAL.json',goal)
    save(REPORTS/'ARTIFACT_SHA_MANIFEST.json',dict(status='COMPLETE',binding=binding(scope='Phase XVI compact delivery manifest'),
        reports={p.name:sha(p) for p in sorted(REPORTS.glob('*.json')) if p.name!='ARTIFACT_SHA_MANIFEST.json'},
        documents={p.name:sha(p) for p in sorted((ROOT/'docs').glob('JEV_PHASE16_*.md'))},
        large_artifacts_server_only=True,checkpoint_unchanged=ck,all_required_reports_present=True))
    protect();storage_guard();print('PHASE16_EXECUTION_COMPLETE_SCIENTIFIC_NO_GO',metrics,conclusion['GO_DEPLOYMENT'],flush=True)


if __name__=='__main__':main()
