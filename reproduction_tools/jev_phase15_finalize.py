"""Conditional bounded scientific failure delivery; never fabricate formal runs."""
import argparse
import collections
from jev_phase15_common import *
from jev_phase15_pilot_finalize import main as pilot_finalize


def document(name,body):
    path=ROOT/'docs'/name;assert name.startswith('JEV_PHASE15_')
    path.write_text(body.strip()+'\n')


def main(version=3):
    protect();assert version==3
    originals=OUT/'finalization_inputs';originals.mkdir(exist_ok=True)
    frozen_refs={}
    for p in REPORTS.glob('*.json'):
        target=originals/p.name
        if not target.exists():shutil.copy2(p,target)
        frozen_refs[(str(p),sha(target))]=ref(target)
    def historical(value):
        if isinstance(value,dict):
            result={k:historical(v) for k,v in value.items()}
            key=(value.get('path'),value.get('SHA256'))
            if key in frozen_refs:
                result.update(frozen_refs[key],original_report_path=value['path'])
            return result
        if isinstance(value,list):return [historical(v) for v in value]
        return value
    for v in [1,2,3]:pilot_finalize(v)
    pilots=read(REPORTS/'PILOT_RESULTS.json');assert len(pilots['versions'])==3
    assert all(not r['full_pilot_qualified'] and not r['pending_native_videos'] for r in pilots['versions']), 'qualified pilot requires actual formal matrix, not this failure delivery'
    official=read(OUT/f'pilot_pooled_live_v{version}/RESULT.json');assert official['status']=='COMPLETE'
    efficiency=read(OUT/f'latency_v{version}/RESULT.json');assert efficiency['status']=='COMPLETE'
    for name,value in [('OFFICIAL_MATLAB_RESULTS.json',official),('EFFICIENCY.json',efficiency)]:save(REPORTS/name,value)
    queries=[];collections_native=[];labels=collections.Counter()
    for v in [1,2,3]:
        value=read(OUT/f'all_query_audit_v{v}/RESULT.json');assert value['status']=='COMPLETE';queries.append(value)
        save(REPORTS/f'ALL_QUERY_RELIABILITY_V{v}.json',value)
        if v<3:
            corrected=read(OUT/f'all_query_selected_probability_v{v}/RESULT.json');assert corrected['status']=='COMPLETE'
            save(REPORTS/f'SELECTED_ACTION_RISK_V{v}.json',corrected)
    for video in TRAIN:
        result=read(OUT/'onpolicy_pilot_dataset_v3_r2'/f'video{video:02d}/RESULT.json');assert result['status']=='PASS'
        collections_native.append(result);labels.update(result['counts'])
    checkpoints=[read(OUT/f'training_full_payload_v{v}/F_full/seed20261009/pilot/RESULT.json')['checkpoint'] for v in [1,2,3]]
    source=binding(seed=20261009,checkpoints=checkpoints,dataset=[r['DATASET'] for r in collections_native],
        evaluator='all bounded TRAIN audits, actual full live DEV TrackEval, official native MATLAB, synchronized runtime',
        scope='bounded scientific NO_GO after three evidence-driven versions; formal matrix conditional and not eligible')
    basic=dict(binding=source,Full24=False,heldout_20_21_22='SEALED',official_TEST=False)
    original_labels=historical(read(REPORTS/'COMMITMENT_LABEL_AUDIT.json'))
    save(REPORTS/'COMMITMENT_LABEL_AUDIT.json',dict(original_labels,original_frozen_corpus_audit=original_labels,
        version2_recent_certificate_audit=ref(REPORTS/'RECENT_OWNER_LABEL_IDENTIFIABILITY.json'),
        version3_own_policy_counts=dict(labels),version3_own_policy_collections=[ref(OUT/'onpolicy_pilot_dataset_v3_r2'/f'video{v:02d}/RESULT.json') for v in TRAIN],
        final_versions_retained=True))
    unknown=historical(read(REPORTS/'UNKNOWN_RELIABILITY_AUDIT.json'))
    save(REPORTS/'UNKNOWN_RELIABILITY_AUDIT.json',dict(unknown,initial_label_audit=unknown,
        final_all_query_audits=[ref(REPORTS/f'ALL_QUERY_RELIABILITY_V{v}.json') for v in [1,2,3]],
        selected_action_probability_corrected_audits=[ref(REPORTS/f'SELECTED_ACTION_RISK_V{v}.json') for v in [1,2]],
        conditional_accuracy_is_not_all_query_accuracy=True,UNKNOWN_is_neither_negative_nor_positive=True,
        posterior_feature_attribution=ref(REPORTS/'POSTERIOR_FEEDBACK_ATTRIBUTION.json')))
    contract=historical(read(REPORTS/'NATIVE_STATE_CONTRACT.json'))
    save(REPORTS/'NATIVE_STATE_CONTRACT.json',dict(contract,original_contract=contract,
        actual_learned_policy_restore_payloads_bitwise=sum(r['persistent_restore_payloads_exact'] for r in collections_native),
        learned_full_inputs_logits_IDs_and_final_memory_exact=all(r['full_inputs_logits_IDs_and_commitment_memory_exact'] for r in collections_native),
        learned_restore_sources=[ref(OUT/'onpolicy_pilot_dataset_v3_r2'/f'video{v:02d}/RESULT.json') for v in TRAIN],
        safety_gate_is_separate_and_FAILED=True))
    structural=read(REPORTS/'STRUCTURAL_TESTS.json');structural.update(native_full_loop='PASS_ACTUAL_512_OLD_AND_512_LEARNED_RESTORE',
        full_loop_contract=ref(REPORTS/'NATIVE_STATE_CONTRACT.json'));save(REPORTS/'STRUCTURAL_TESTS.json',structural)
    repair=read(REPORTS/'MATLAB_AGGREGATION_PATH_REPAIR.json')
    repair.update(status='REPAIRED_OFFICIAL_EVALUATION_COMPLETE',official_result=ref(OUT/f'pilot_pooled_live_v{version}/RESULT.json'),
        repaired_execution_log=ref(OUT/'MATLAB_AGGREGATION_PATH_REPAIRED.log'))
    save(REPORTS/'MATLAB_AGGREGATION_PATH_REPAIR.json',repair)
    notrun=dict(status='NOT_RUN_PILOT_SAFETY_GATE_FAILED',metrics=None,actual_updates=0,
        reason='Frozen protocol forbids formal training and new controls before a reliable F pilot; all three completed pilots fail real native safety.')
    save(REPORTS/'FORMAL_TRAINING_RESULTS.json',dict(**basic,**notrun,formal_seeds=SEEDS,formal_budget_per_run=20000,
        planned_arms=['A_original','B_continuity','C_commitment','D_fixed','E_set','F_full'],
        bounded_Tiny_and_Pilot_actual_updates=3*(128+1500),extra_retained_legacy_engineering_Tiny_updates=128,
        all_new_training_updates_including_engineering=3*(128+1500)+128,last_checkpoints=checkpoints,pilots=ref(REPORTS/'PILOT_RESULTS.json')))
    old_online=read(ROOT/'reports/JEV_PHASE14/ONLINE_VALIDATION.json');old_matlab=read(ROOT/'reports/JEV_PHASE14/OFFICIAL_MATLAB_RESULTS.json')
    save(REPORTS/'FAIR_BASELINE_RESULTS.json',dict(**basic,**notrun,
        same_DATA_same_compute_new_A_to_F=None,controlled_Q4_and_option_retraining_ablations=None,
        prior_phase14_reference=ref(ROOT/'reports/JEV_PHASE14/ONLINE_VALIDATION.json'),
        prior_phase14_formal_means=old_online['seed_summary']['formal'],
        prior_phase14_official_reference=ref(ROOT/'reports/JEV_PHASE14/OFFICIAL_MATLAB_RESULTS.json'),
        prior_phase14_official_formal_means=old_matlab['seed_summary']['formal'],
        prior_models_not_relabelled_as_new_fair_controls=True,independent_Jev_advantage_proven=False))
    save(REPORTS/'ON_POLICY_STABILITY.json',dict(**basic,status='DIAGNOSTIC_OWN_STATE_COLLECTION_AND_PILOT_COMPLETE_FORMAL4K_NOT_ELIGIBLE',
        formal_onpolicy_4k_metrics=None,formal_onpolicy_actual_updates=0,formal_rounds=0,
        diagnostic_collector_policy=checkpoints[1],diagnostic_one_collection_pass_per_TRAIN=True,
        diagnostic_pilot_training_updates=1500,diagnostic_collection_counts=dict(labels),
        collector_never_teacher_forced=True,final_pilot=pilots['versions'][-1],
        all_versions_and_original_failed_artifacts_preserved=True,qualifying_matched_A_to_F_4k_adaptation_not_run=True))
    online=[];propagations=[];examples=[]
    for video in DEV:
        path=OUT/f'pilot_online_v{version}/F_full_seed20261009/live'/f'video{video:02d}/RESULT.json';r=read(path)
        assert r['status']=='COMPLETE' and r['live_images'] and r['actual_mutated_state_online']
        online.append(dict(video=video,result=ref(path),frames=r['frames'],metrics=r['strict_online_metrics'],risk=r['native_risk'],
            complete_native_state_SHA256=r['final_identity_and_commitment_memory_SHA256'],perception_checks=r['live_frontend_vs_frozen_cache_four_payload_checks']))
        p=read(r['error_propagation']['path']);propagations.append(dict(video=video,artifact=r['error_propagation'],counts=p['counts'],
            episodes=len(p['episodes']),right_censored=sum(e['right_censored'] for e in p['episodes']),scope=p['scope'],semantics=p['semantics']))
        examples.extend(dict(video=video,**item) for item in sorted(p['episodes'],key=lambda e:e['observed_frames'],reverse=True)[:3])
    save(REPORTS/'ONLINE_VALIDATION.json',dict(**basic,status='COMPLETE_SINGLE_LAST_PILOT_LIVE_DIAGNOSTIC',version=version,
        seed=20261009,cases=online,pooled_raw_TrackEval=official['pooled_raw_TrackEval'],raw_predictions=official['raw_predictions'],
        full_video_live_frontend_and_native=True,actual_GT_used_for_scoring_only=True,raw_filter_or_GT_renumbering=False,
        formal_three_seed_metrics=None,independent_test=False,formal_tracking_GO_claimed=False,full_FPS=None))
    save(REPORTS/'IDENTITY_ERROR_PROPAGATION.json',dict(**basic,status='COMPLETE',cases=propagations,longest_observed_examples=examples,
        gaps_and_video_end_are_censored=True,continued_wrong_owner_after_gallery_mixing_counted=True,
        current_GT_unknown_unassessed=True,identity_owner_is_fixed_first_observed_GT_not_canonical_GT_mapping=True,
        actual_bootstrap_owner_seeds_included=True,duration_semantics_contract=ref(OUT/'error_duration_contract/RESULT.json'),
        physical_time_seconds=None,physical_frame_rate_not_independently_verified=True))
    exposure=read(ROOT/'reports/JEV_PHASE14/PRETRAIN_EXPOSURE_AUDIT.json')
    save(REPORTS/'GENERALIZATION_QUALIFICATION.json',dict(**basic,status='NO_GO_ABSOLUTE_UNSEEN_FULL_SYSTEM',
        prior_exposure_audit=ref(ROOT/'reports/JEV_PHASE14/PRETRAIN_EXPOSURE_AUDIT.json'),
        Stage1_actual_training_videos=exposure['Stage1_actual_training_videos'],
        historically_reused_DEVELOPMENT=True,association_training_DEV_disjoint=True,
        unseen_absolute_generalization_proven=False,clean_frontend_retraining_metrics=None,external_transfer_metrics=None,
        reason='Known Stage1 trained on all24; clean/external expansion conditional on scientific qualification. No sealed20/21/22 inference or inspection.'))
    adverse=[dict(version=v['version'],**c) for v in pilots['versions'] for c in v['adverse_native_cases']]
    save(REPORTS/'NATIVE_UNSAFE_EVENTS.json',dict(**basic,status='COMPLETE_ALL_FAILURES_RETAINED',events=adverse,
        counts_are_horizon_events_not_independent_clusters=True,
        exact_native_traces_available_by_ref=True,ambiguous_or_mixed_history_not_assumed_wrong=True))
    save(REPORTS/'FINAL_GO_NO_GO.json',dict(**basic,status='SCIENTIFIC_NO_GO',execution_deliveries='COMPLETE_BOUNDED_CONDITIONAL_PROTOCOL',
        scientific_success=False,GO_TRACKING=False,GO_JEV_INDEPENDENT_VALUE=False,GO_DEPLOYMENT=False,
        full_pilot_qualified=False,three_versions_attempted=True,pilots=ref(REPORTS/'PILOT_RESULTS.json'),
        diagnostic_metrics=official['pooled_raw_TrackEval'],diagnostic_CVIDF1=official['CVIDF1'],diagnostic_CVMA=official['CVMA'],
        formal_metrics=None,independent_value_metrics=None,
        gates=dict(native_structure='PASS',conditional_commitment_mechanism='PASS',learned_native_safety='FAIL',
            formal_tracking='NOT_ELIGIBLE',independent_architecture='NOT_ELIGIBLE',deployment='FAIL' if not efficiency['runtime_thresholds_pass'] else 'QUALITY_NOT_ELIGIBLE'),
        current_hard_barriers=['Perfect/near-perfect certified continuation is not all-query safety.',
            'Actual new policy changes histories and can mix identities despite reduced IDSW.',
            'No clean compatible certificate for many UNKNOWN queries; repeated ambiguous/mixed IDs cannot be declared negative.',
            'Global cross-camera identity quality and error duration require complete raw evaluation; temporal IDSW alone cannot certify success.'],
        posterior_feedback_causal_explanation_rejected=True,all_formal_and4k_unrun_metrics_null=True,
        posterior_negative_attribution_scope='v1 checkpoint, all24 frozen TRAIN windows only; not an extrapolation to later weights or unseen states',
        fair_ordinary_network_superiority_or_equivalence_not_tested=True,prior_failures_weights_and_seeds_preserved=True,
        recommended_next_qualified_research='Repair identifiable compatibility/recovery for no-clean-candidate states, collecting independently certified correction events; investigate candidate/lifecycle constraints before more Q4 capacity or epochs. Freeze a new budget/protocol before further versions.'))
    final=historical(read(REPORTS/'FINAL_GOAL.json'));final.update(status='BOUNDED_EXECUTION_COMPLETE_SCIENTIFIC_NO_GO',scientific_status='SCIENTIFIC_NO_GO',
        result=ref(REPORTS/'FINAL_GO_NO_GO.json'),objective_success=False,formal_training_conditional_NOT_RUN=True)
    save(REPORTS/'FINAL_GOAL.json',final)
    tiny=read(REPORTS/'TINY_RESULTS.json');rows=[]
    for p in pilots['versions']:
        a=p['assessment'];rows.append(f"| {p['version']} | {p['actual_updates']} | {a['safe_continuation_accuracy']} | {a['necessary_correction_accuracy']} | {a['counts'].get('kind2_support',0)} | {len(p['adverse_native_cases'])} | NO_GO |")
    metrics=official['pooled_raw_TrackEval'];metric_text=json.dumps(metrics,ensure_ascii=False,indent=2)
    document('JEV_PHASE15_TRAINING_AND_ABLATIONS.md',f'''# Phase XV Training and Ablations

三版都完成独立 Tiny128 和 Pilot1500，seed20261009，matching Phase XIV20k 初始化，固定 LAST。Pilot 不接着 Tiny 训练。固定 AdamW lr0.0003/wd0.01/batch4，权重 WHO1/Availability0.5/Trust0.5/Commitment1/whole-payload Assignment0.2/Risk0.5。

| 版本 | Pilot updates | 保留区安全延续准确率 | 必要纠错准确率 | 必要纠错 support | 不安全 horizon 事件 | 资格 |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

V1 实现 Q4 和可靠性，但原保留区没有必要纠错认证；认证准确率不代表所有查询。V2 增加 actual recent3-frame 的局部不安全承诺认证，全球混合 WHO 标签仍 UNKNOWN；新增训练12、保留5个纠错标签。V3 在 V2 原生错误和状态变化证据后，采集它自己真实产生的完整 TRAIN 历史，与原 Multi corpus 混合，固定相同1500 updates，不改变层数、学习率或损失权重。

三版 Tiny 的固定四个代表 payload 均来自原 frozen-Multi corpus，V1和V3相同输入/初始化可产生相同loss与权重；不能把它们当作独立数据复现。V3的新 own-state 干预发生在 Pilot；1500 steps共6000个采样payload中2958来自实际own-v2状态、3042来自原Multi，7个审计纠错样本最终做对5个。Tiny loss下降只证明局部梯度可学，不证明新状态泛化或原生安全。

另保留旧工程 Tiny128：旧联合损失丢掉未知检测行，修复后在新命名空间重跑。总计新梯度 updates5012，其中正式20k=0；旧文件、旧权重和失败日志仍在服务器。训练数据仅12/13/14/16，reserved temporal block不送入优化器，DEV17/18/19仅作诊断，20/21/22封存。

A–F 三种子20k、公平架构消融和 matched4k 正式 on-policy 都依赖可靠 F Pilot。三版都未通过原生安全门，因此这些条件阶段 NOT_RUN、指标 null。历史 Fixed/Multi/Set 仅作历史参考，不能冒充本次同监督/同算力对照；本次没有证明普通网络与 Jev 等价或更优，也没有证明 Jev 独立优势。

反馈通道归因：V1 same checkpoint/prefix/RNG 的24个实际 native 窗口屏蔽 purity/posterior-available 输入，实际 ID 变化数0。该阴性归因只排除该通道对这些V1窗口的动作影响，不外推到后两版权重、其他状态或完整视频；不能将“分布上未覆盖”推断成已证明的因果原因。三版各自 all-query audit、全部不安全 horizon 和 raw轨迹均由 JSON SHA绑定。
''')
    last=queries[-1]['results']['F_full']['counts']
    document('JEV_PHASE15_UNKNOWN_RELIABILITY.md',f'''# Phase XV UNKNOWN and Reliability

UNKNOWN 保留为合法推理候选，不自动作负例或正例。WHO 的已认证纯净身份兼容性、具体 Commit 的可安全延续性、全局 history purity、预测 safety 与 uncertainty 分开。

V1 all-query audit:3140个当前 GT已知查询中1309选择 UNKNOWN，995选择全局混合历史；WHO认证查询仅1822。认证子集接近满分不能代表其余困难查询的风险。V2 在3160个已知查询中1310选择UNKNOWN，990选择混合历史；不采用未认证情况的猜测真值。两版各自选择认证错误11和12，UNKNOWN 不计为已认证错误。

V3 全部保留查询计数（其审计 corpus 与前两版不同，不能直接把绝对计数当改进）：

```json
{json.dumps(last,ensure_ascii=False,indent=2)}
```

Risk-coverage 的 coverage 分母采用全部当前 GT已知查询，certified-selection risk仅在被选候选已有认证时计算，UNKNOWN数量另列。当前 GT未知的检测另列 unassessed。置信度是实际联合分配选中动作的softmax概率；不能用单行最大分替代。在V1/V2分别有201/157个查询的联合动作不同于单行argmax；旧最大概率口径保留，SELECTED_ACTION_RISK_V1/V2修正副本分别绑定不变的checkpoint，没有额外SGD。概率未经独立校准，高置信UNKNOWN不等于正确。

V2 recent3-frame 认证只给“该候选当前局部承诺不安全”提供证据；未知/缺帧、跨帧不连续、最近属于当前人、没有纯净兼容替代，都不制造纠错标签。WHO对混合历史仍UNKNOWN。更一般的无纯净候选状态不能随意补充正确候选或宣称 DEFER一定安全：P2 DEFER实际常造成假出生和错误混合。

完全保留 annotation，重复 GT身份仅使离线认证/风险观察 UNKNOWN；TrackEval及官方 MATLAB仍读取原始GT。训练不能用DEV结果挑门槛。真实完整视频风险观察、错误时长与CVIDF1独立报告。
''')
    full_rows='\n'.join(f"| video{c['video']:02d} | {c['frames']} | {json.dumps(c['metrics'],ensure_ascii=False)} |" for c in online)
    longest='\n'.join(f"- video{x['video']:02d}, camera{x['view']}, GT{x['GT_OFFLINE_ONLY']}, frame{x['start']}–{x['end']}: {x['observed_frames']}个实际错误观测帧，right_censored={x['right_censored']}。" for x in examples)
    runtime='\n'.join(f"- Trial{t['repeat']}: sceneFPS {t['full_two_camera_scene_FPS']:.3f}, total native Stage2 p95 {t['total_native_Stage2_ms']['p95']:.3f}ms。" for t in efficiency['trials'])
    document('JEV_PHASE15_FINAL_RESEARCH_REPORT.md',f'''# WWW / JEV Phase XV Final Research Report

**SCIENTIFIC_NO_GO。** 三版有界训练和真实原生诊断均已完成；合法 Commitment 在部分前缀存在纠错机会，但学习式控制尚未满足原生身份安全门。本次不是 WWW科研成功，也没有独立 Jev 架构优势结论。正式20k三种子和 matched4k 的条件不满足，未运行指标为null。

## 完成的证据链

P0 从0dc9607冻结 V–XIV，332个checkpoint原始hash以及83个受保护报告保留。P1 完成15个旧模型/视频实际原生重放，含 Multi三个种子和seed20261009的Fixed/Set，inputs/scores/IDs/最终状态完全一致。最坏种子824个CLEAR IDSW：pure fragment158、polluted history505、history corrective109、wrong existing merge30、false birth5、recovery1、ambiguous16；其中277个有合法纯净旧候选，只占33.62%，不能把所有切换都罚掉。

P2 冻结46个前缀、5个动作、2个后续策略，共460个声明分支；430实际执行、30个认证替代不可用。πMulti与πFixed分别有8/18、12/18独立安全延续获益cluster，达到预冻结机制可行性门。这是条件性实际因果证据，不是完整视频收益或架构优越性。KEEP目标及ALT兼容性由过去观测的GT离线认证，仅用于研究干预，不代表部署模型已识别该动作；任何动作仍受原生合法候选限制。TRAIN14原始GT重复导致严格CLEAR失败，另保留6个隔离的permissive诊断，排除严格门；原始失败不覆盖。DEFER多数窗口增加出生，强制KEEP也有混合增多窗口。

建立 GT-free Persistent CommitmentState，明确WHO vs具体native ID承诺，动态candidate-conditioned Q4，独立purity/safety/uncertainty，真实整个检测集合的Hungarian loss，原生 Gallery/Bank/ID/REACT/START_NEW不变。原 Multi四个完整TRAIN预测完全不变，512 payload恢复bitwise一致；另用第三版收集器验证512个学习式、带posterior状态payload恢复。Alias关闭；没有GT canonical renumbering。

## 有界修复与结果

详见 `JEV_PHASE15_TRAINING_AND_ABLATIONS.md` 和 `PILOT_RESULTS.json`。V1 缺必要纠错审计支持；V2 建立过去真实连续片段证据，必要纠错5/5但仍出现native污染；V3 使用V2自身完整TRAIN mutated-state历史，修复训练状态分布，并再次固定Tiny128/Pilot1500。全部版本及失败保留，没有挑最好checkpoint、丢最差种子、调DEV阈值或增加epochs。

V1相同权重、同prefix/RNG，仅屏蔽过去可靠性反馈的24个干预窗口没有任何实际ID变化；该证据只否定此通道在这些V1窗口的动作解释，不能覆盖V2/V3或未观察状态。原全局混合状态中大量查询缺少可认证纯净兼容候选；在 UNKNOWN上保持强延续可能延长身份混合，靠把UNKNOWN当负例无法可信解决。

具体反例：TRAIN12 frame3/camera1/row6前缀，V3在H32把IDSW减少4，却使新跨GT混合增加2、固定owner错误观测增加36、污染历史写入增加71。第一步仍延续可认证正确的ID9，后续策略自行产生的不安全提交才改变历史。它说明第一步延续正确及IDSW下降都不能保证闭环身份正确。这些数值是配对窗口差值；不同H及相邻窗口不当独立样本，精确输入与后续raw轨迹见NATIVE_UNSAFE_EVENTS.json的源文件引用。

出生口径：冻结的false_birth代理统计已出现GT再次START_NEW，可能包含必要恢复；false_split_birth另要求仍有可认证正确的合法旧候选。两者均保留，不把所有恢复出生自动判错。三版均存在实际新混合增加，故原生安全失败不只依赖该保守出生代理。

## 完整视频与官方指标

预先固定第三版LAST、seed20261009，在 DEV17/18/19完整当前图像→真实Stage1/VFCE→实际native Gallery/Bank状态运行。没有截断、短轨过滤、理想化teacher state或GT重编号。以下是诊断，不是合格正式三种子比较。

| 视频 | 完整scene frames | raw TrackEval |
|---|---:|---|
{full_rows}

六个camera的 pooled raw TrackEval：

```json
{metric_text}
```

原生官方MATLAB：**CVIDF1 {official['CVIDF1']:.6f}, CVMA {official['CVMA']:.6f}**。使用未修改官方函数和已认证MEX；从每scene sequential身份计数求CVIDF1，从interleaved CLEAR计数求CVMA，不能平均video分数。历史 Phase XIV Multi seed09 HOTA73.476051/IDSW824/CVIDF187.195675；历史Fixed seed09 HOTA71.891351/IDSW206/CVIDF183.939320。历史三种子均值或不同阶段初始化不能替代本次公平A–F实验。

## 真正错误持续时间

在实际提交ID上固定第一次可观察GT为身份owner，包含真实最大摄像头bootstrap；后续混合仍累计错误。gap和视频结束右删失，不把缺帧当纠正；当前GT不能可靠匹配则unassessed。计量单位是实际错误观测帧，未验证真实采样帧率，不把它猜成秒。该offline observed-confusion定义与CLEAR/全局IDF1不同，完整episodes保存在服务器。

{longest}

## 真实速度与泛化

独立关闭GT/journals/evaluator，以真实图像输入重复3次DEV17前256scene frames，包括IO、detector、VFCE、history、policy、assignment、Gallery/Bank和posterior提交；此为共享GPU上的前缀运行，完整视频未审计的FPS为null，不用缓存FPS冒充。实际测量：

{runtime}

部署门还要求可靠的tracking质量；当前不具备部署资格。Stage1已在全部24个视频训练，DEV被历史研究使用。本次是关联控制器scene-disjoint训练和已使用DEV诊断，不能宣称绝对未见全系统泛化。20/21/22、官方TEST仍封存，Full24未启动，未把不合格外部WILDTRACK前缀冒充迁移证明。

## 科学结论与剩余阻塞

1. WHO正确候选集合与具体身份Commitment不同，存在合法可受益前缀；该机制可行性得到了原生配对支持。
2. 高认证延续准确率未足以解决全部查询的安全、错误后恢复及跨摄像头统一身份，三版均未达到完整native Pilot安全门。
3. 新增Q4和可靠性模块是否优于普通continuity或Fixed/Set同监督网络，本次没有资格开展正式公平对照，结论为未证实；不能声称等价，也不能声称Jev胜出。
4. 下一步必须先让无纯净兼容候选的状态具备可识别的纠错/恢复证据，区分必要出生与反复碎裂，检查视觉候选及原生生命周期约束。需新冻结预算和独立可认证纠错事件，再扩展架构或长训。对未知历史一律加负标签、靠总IDSW下降宣告成功或继续盲增epochs，都不构成当前阻塞的解决。

代码与紧凑JSON在独立 Phase XV branch；模型、数据、原始轨迹、失败和hash绑定文件留服务器。所有原始V–XIV研究资产仍保留，完整最终保护校验见FINAL_PRIOR_INTEGRITY.json。阶段条件及精确失败事件分别见FINAL_GO_NO_GO.json与NATIVE_UNSAFE_EVENTS.json。
''')
    architecture=ROOT/'docs/JEV_PHASE15_PERSISTENT_IDENTITY_ARCHITECTURE.md'
    body=architecture.read_text().replace('Version 1 structural tests passed for state snapshots, restore, stale recycling, cross-camera sharing, candidate permutation and candidate-specific values. Complete TRAIN native parity and data qualification are pending. No optimizer update has yet been run. The research gates and loss weights remain the frozen preregistration.',
        'Structural contracts passed for snapshots, restore, stale recycling, cross-camera sharing, candidate permutation and candidate-specific values. Full TRAIN collection verified 512 original-policy and 512 learned-policy restored payloads bitwise, including final memory. Three independent Tiny128/Pilot1500 versions and an extra retained legacy Tiny128 completed 5012 optimizer updates. All three pilots failed native safety; formal20k, controlled architecture retraining and matched4k remain unqualified and NOT_RUN. Full live DEV and native official MATLAB evaluation are complete diagnostics, not formal three-seed success. The research gates and loss weights remain the frozen preregistration.')
    architecture.write_text(body)
    extra='''\n## Final implementation qualification\n\nWHO certificate and concrete commitment certificate are separate. V2 adds `commit_known_options` only for a reliably wrong recent committed segment; mixed WHO labels stay UNKNOWN. V3 collects the frozen second pilot's real full TRAIN histories, including actual stored posterior inputs, and mixes those with the immutable frozen-Multi corpus. Neither repair changes native candidate generation, Gallery/Bank, one-to-one capacity or IDs. The two-channel posterior intervention changed zero actual IDs; it is a retained negative attribution. Three bounded pilots fail native safety; this is an implemented failed research version, not a validated tracking or architecture advantage.\n'''
    if '## Final implementation qualification' not in architecture.read_text():architecture.write_text(architecture.read_text()+extra)
    required=['FINAL_GOAL','PHASE14_FROZEN_EVIDENCE','SWITCH_EVENT_LEDGER','SEED20261009_FAILURE_ATLAS','COMMITMENT_NATIVE_CAUSAL_BRANCHES','COMMITMENT_FEASIBILITY_GO_NO_GO','COMMITMENT_LABEL_AUDIT','UNKNOWN_RELIABILITY_AUDIT','NATIVE_STATE_CONTRACT','STRUCTURAL_TESTS','TINY_RESULTS','PILOT_RESULTS','FORMAL_TRAINING_RESULTS','FAIR_BASELINE_RESULTS','ON_POLICY_STABILITY','ONLINE_VALIDATION','OFFICIAL_MATLAB_RESULTS','EFFICIENCY','GENERALIZATION_QUALIFICATION','FINAL_GO_NO_GO']
    delivery=dict(status='COMPLETE_BOUNDED_SCIENTIFIC_NO_GO',binding=source,
        required_results=[ref(REPORTS/(n+'.json')) for n in required],
        required_documents=[ref(ROOT/'docs'/n) for n in ['JEV_PHASE15_RESEARCH_GOAL.md','JEV_PHASE15_PERSISTENT_IDENTITY_ARCHITECTURE.md','JEV_PHASE15_CAUSAL_IDENTITY_COMMITMENT.md','JEV_PHASE15_UNKNOWN_RELIABILITY.md','JEV_PHASE15_TRAINING_AND_ABLATIONS.md','JEV_PHASE15_FINAL_RESEARCH_REPORT.md']],
        all_large_weights_data_predictions_and_failed_logs_server_only=True,
        preserved_prior_bytes=ref(OUT/'final_integrity/RESULT.json'),
        actual_failure_traces=ref(REPORTS/'NATIVE_UNSAFE_EVENTS.json'),
        initial_reports_preserved=[ref(p) for p in sorted(originals.glob('*.json'))],
        no_protected_prior_files_modified=True,task_delivery_is_not_scientific_success=True)
    save(REPORTS/'DELIVERY_AUDIT.json',delivery)
    print('PHASE15_BOUNDED_FINAL_DELIVERY',metrics,official['CVIDF1'],official['CVMA'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',type=int,default=3);a=p.parse_args();main(a.version)
