"""Freeze MATCH-only scope and matrix before any new training or metric result."""
from jev_oneweek_common import *


def main():
    protect()
    target = REPORTS/'PREREGISTRATION.json'
    assert not target.exists(), 'Do not overwrite a frozen protocol'
    protocol = dict(
        status='FROZEN_BEFORE_NEW_SPRINT_RESULTS', base=BASE,
        scope='Controlled current association learning; not future safe-action learning',
        architecture='Phase XIV algorithm + Phase XV native interface; no hard/soft memory control',
        TRAIN_authorized=TRAIN, new_gradient_videos=[12,13,14],
        controller_audit_video=16, development=DEV,
        split=dict(unit='video plus GT-person dependency groups; temporal blocks are correlated diagnostics',
                   inherited_backbone_and_initializers_exposure='All24TRAIN Stage1; previous Stage2 TRAIN12/13/14/16; repeated DEV17/18/19',
                   video16_role='No gradients for new sprint modules; exposed to inherited initializers, not independent system heldout',
                   person_namespace='Treat GT persons within each video as dependent; audit cross-video scene/person overlap before claiming independence',
                   known_prior_temporal_audit='(frame//64)%5==4; preserve annotations, do not rename it independent validation'),
        model_selection='Last fixed-budget checkpoint; use only controller audit16 for preregistered engineering/pilot gates. No DEV tuning, best-seed selection or best-checkpoint search',
        seeds=SEEDS, training=dict(tiny_updates=128, pilot_updates=1000, formal_total_updates=4000,
                                  optimizer='AdamW', learning_rate=0.0001, weight_decay=0.01,
                                  batch_payloads=4, frozen_common_initializer=True,
                                  trainable_new_parameter_target=500000,
                                  ordinary_controls_same_supervision=True,
                                  natural_sampling='Uniform video, then 64-frame block, then payload. Additional natural hard stratum has separately recorded weight; no copied rare safety events',
                                  synthetic_corruption_in_primary=False),
        supervision=dict(current_choice='Certified known past owners with current TRAIN detection GT; UNKNOWN masked',
                         cross_camera='Current GT vs certified other-camera past observations; per-view mixed/duplicate/unmatched history UNKNOWN',
                         stability='Observable predecessor continuity supported by current and past certified identities; not purity alone, not future value',
                         horizon_value_heads=False, learned_MEMORY_WRITE=False, learned_REACT=False),
        candidate_plans=dict(max_count=4, base_always_included=True, generation='Deterministic legal-edge conflicts, native per-camera capacity, GT/未来帧 forbidden',
                             same_set_for_strong_controls=True, same_solver_budget=True,
                             additive_vs_interaction_required=True, zero_initialized_output_parity=True),
        mechanism_factorial=['OriginalScores_OriginalSolve','OriginalScores_ExtraSolve','JEVScores_OriginalSolve','JEVScores_ExtraSolve'],
        extra_solve='Rerunning unchanged Hungarian is a no-op control, not a claimed reassociation benefit',
        comparisons={f'M{i}': name for i,name in enumerate([
            'Original complete GMT/GTA (separate full frontend reference)', 'Same-Stage1 cosine',
            'Fixed threshold', 'State/pair learned thresholds and Binary JEV', 'Fixed/learned linear fusion',
            'Original-rule second association and repeated Hungarian', 'Small MLP and DeepSets',
            'Ordinary Attention/SetTransformer', 'Phase XIII Fixed Question',
            'Phase XIV Multi-question', 'Phase XV V3 frozen diagnostic', 'New JEVTrack'])},
        ablations=['strongest ordinary control','cross_camera_only','stability_only','both',
                   'fixed_question','no_shared_state','additive_edges_only','equal_extra_solver_no_JEV','ordinary_attention_all_evidence'],
        primary_metrics=['HOTA','DetA','AssA','IDF1','MOTA','IDSW','Frag','CVIDF1','CVMA'],
        raw_online_primary=True, future_short_track_filter_secondary_only=True,
        metrics='Frozen TrackEval and official MATLAB; true pooled camera/scene counts, not mean-of-video HOTA',
        attribution='Paired same-state local interventions and separately labelled full mutated-state closed-loop outcomes',
        budget=dict(major_algorithm_versions_max=3, runtime_retention_GiB_max=8,
                    minimum_home_and_data1_free_GiB=30, new_large_downloads=0,
                    runtime_full_restarts_per_method=3, speed_relative_loss_target=0.10),
        gates=dict(engineering='Finite loss/gradients, legality, GT separation, initial native parity, order/dynamic K and full native transitions',
                   pilot='Finite learning and no label-contract violation; association improvements assessed on reused controller audit, no original MASTER safe-action gate reinterpretation',
                   collapse='Record full native failures, retain safe baseline. Never delete a failing seed or relabel UNKNOWN',
                   science='Positive HOTA/AssA/CVIDF1 against strongest fair ordinary control plus protection/IDSW/cost evidence',
                   JEV_independence='All five reviewer questions require separate actual evidence; otherwise NOT_ESTABLISHED',
                   exploratory_HOTA_target_points=0.5),
        scientific_success_not_assumed=True, original_MASTER_safety_gate_unchanged=True,
        official_TEST=False, sealed20_21_22=True, Full24=False,
        conference_dates=dict(abstract='2026-10-18 AoE', full_paper='2026-10-25 AoE',
                              official_source='https://www2027.thewebconf.org/research-track-papers/'))
    save(target, protocol)
    registry=dict(status='EXECUTING_DAY1', max_major_versions=3, current_major_version=1,
                  jobs={k:dict(status='PLANNED', results=None) for k in [
                      'historical_asset_audit','parameter_count','original_GMT_restore',
                      'TRAIN_supervision_audit','initial_parity','tiny_training','pilot_controls',
                      'native_closed_loop','ablations','full_video_efficiency','final_evidence']})
    save(REPORTS/'EXPERIMENT_REGISTRY.json', registry)
    for name in ['MAIN_RESULTS','ABLATIONS','PARAMETERS_AND_FPS','CROSS_CAMERA_ANALYSIS']:
        save(REPORTS/(name+'.json'), dict(status='NOT_RUN', values=None,
             reason='Frozen plan registered; actual execution results will replace this explicitly pending entry'))
    save(REPORTS/'GO_NO_GO.json', dict(status='RUNNING_NOT_ESTABLISHED', engineering=None,
         association=None, JEV_independent_value='NOT_ESTABLISHED', paper_readiness='NOT_READY',
         original_MASTER_goal_achieved=False))
    text='''# JEVTrack 一周科研执行目标

研究跨摄像头证据是否能够提高 Global ID 关联准确性并保持长期身份稳定。采用 Phase XIV 算法与 Phase XV 原生接口，先恢复基线和同条件比较，再验证跨摄像头一致性与可观察身份连续性。真实结果可以否定 JEV 假设。

本周独立 MATCH-only 协议允许已有 TRAIN12/13/14/16 受控关联研究，不构成原 MASTER 长期安全纠错监督门槛通过。新模块梯度限 TRAIN12/13/14，16 仅作已暴露控制器审计；DEV17/18/19 仍是反复使用的开发集。20/21/22、官方 TEST 和 Full24 保持封存。已训练初始化对16及前端对全部 TRAIN 的暴露必须披露。

128-step Tiny → 1000-step Pilot → 至多4000-step正式预算，三个固定种子20261008/09/10。只训练新增模块，所有普通强对照获得相同证据、监督、方案和更新预算。至多三轮主要算法修改，不因第一轮科学失败结束，也不无限扫参。新增留存上限8 GiB，两分区至少保留30 GiB。每个完成阶段推送必要代码、配置、失败、指标和 SHA；模型/图像/轨迹留服务器。

全部实际模型训练、完整 native mutated-state 在线结果、完整视频效率和五个审稿人问题证据都完成后，才形成科学结论。阶段完成不能冒充独立 JEV 贡献成立或原 MASTER 已完成。执行状态见 reports/JEV_ONEWEEK/EXPERIMENT_REGISTRY.json。

官方研究轨日期已核验：[WWW2027 Research Track](https://www2027.thewebconf.org/research-track-papers/)，摘要10月18日、全文10月25日（AoE）。
'''
    (DOCS/'RESEARCH_GOAL.md').write_text(text)
    questions=[
        ('Q1','JEV 与二次关联有什么区别？','原 GMT、规则二次、重复Hungarian、学习二次、新JEV及评分×求解2×2；额外求解数、候选范围和历史分开归因'),
        ('Q2','为什么固定或可学习阈值不够？','固定、state阈值、pair阈值、Binary JEV、相对候选关系；同合法集与历史，检查决策能否被阈值重现'),
        ('Q3','为什么融合、MLP 或普通 Attention 不够？','固定/线性融合、MLP、DeepSets、Attention、Fixed、Multi；相同全部证据/监督/更新/求解，披露继承共同初始化'),
        ('Q4','增加多少参数和成本？','真实加载与forward-active计数、替换GTA净变化，按同GPU/真实双相机完整视频三次重启分解参数/FLOPs/显存/scene FPS/p50/p95'),
        ('Q5','是否符合 JEV 思想？','State一次、两个可独立验证Question、动态Options、概率不确定性与共享计算；固定/单问题/无共享/无Reader/无cross/无stability/顺序/数量/敏感性/校准')]
    qtext='# 五个审稿人问题：所有实验强制检查\n\n每次计划登记须引用下表问题、同条件契约、结果和限制。Attention或多Head的存在不能替代独立价值证据。\n\n| 问题 | 必需实验 | 当前状态 |\n| --- | --- | --- |\n'
    for key,label,design in questions:qtext+=f'| {key}：{label} | {design} | NOT_RUN |\n'
    (DOCS/'REVIEWER_FIVE_QUESTIONS.md').write_text(qtext)
    for name in ['HISTORICAL_EVIDENCE_AUDIT','METHOD_AND_INNOVATIONS','BASELINE_PARITY','FINAL_REPORT']:
        (DOCS/(name+'.md')).write_text(f'# {name}\n\n状态：NOT_RUN / 正在执行。未运行指标保持 null，尚无本周独立贡献结论。\n')
    print('ONEWEEK_PROTOCOL_FROZEN')


if __name__ == '__main__':
    main()
