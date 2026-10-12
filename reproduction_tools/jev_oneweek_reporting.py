"""Refresh reviewer-facing evidence from completed artifacts, never impute results."""
import collections
import numpy as np
from jev_oneweek_common import *

METRICS=['HOTA','AssA','DetA','IDF1','MOTA','IDSW','Frag','CVIDF1','CVMA']
METHODS={'M1':['cosine'],'M2':['fixed_threshold'],'M3':['threshold','state_threshold'],
         'M4':['linear','fixed_fusion'],'M5':['extra_hungarian','base_rule'],
         'M6':['mlp','deepsets'],'M7':['attention'],'M8':['XIII_fixed_reference'],
         'M9':['parent'],'M10':['XV_V3_reference'],'M11':['jev']}
ABLATIONS={'A0':['mlp','deepsets','attention','linear','state_threshold','threshold'],
           'A1':['cross_only'],'A2':['stability_only'],'A3':['jev'],
           'A4':['fixed_question'],'A5':['no_shared_recompute'],'A6':['additive','edge_single'],
           'A7':['extra_hungarian','base_repeat'],'A8':['attention'],
           'no_cross_evidence':['no_cross_evidence'],'no_stability_evidence':['no_stability_evidence'],
           'no_question_reader':['no_question_reader']}


def refresh():
    protect()
    from jev_oneweek_results import main
    main(quiet=True)
    current=read(REPORTS/'FORMAL_CURRENT_RESULTS.json');arms=current['arms']
    previous=read(REPORTS/'MAIN_RESULTS.json');rows={'M0':previous['rows']['M0']}
    for row,names in METHODS.items():
        rows[row]=dict(status='COMPLETE' if all(len(arms.get(n,{}).get('cases',[]))==(1 if n in ['cosine','XV_V3_reference'] else 3) for n in names) else 'PARTIAL_OR_NOT_RUN',
                      methods={n:arms[n] for n in names},
                      comparison_scope='Common Stage1 VFCE1024 and native lifecycle; learned incremental controls share frozen XIV initialization',
                      original_full_GMT_same_common_frontend=False)
    rows['M1']['single_seed_reason']='Predeclared deterministic cosine policy: no learned controller seed'
    rows['M10']['single_seed_reason']='Only the real frozen XV V3 seed09 pilot exists; missing seeds not fabricated'
    rows['M8']['reference_scope']='Frozen old independently trained Fixed Question, not equal incremental training budget'
    rows['M5']['reference_scope']='Native XIV-score rule/repeated-solver controls; not the original complete GMT legacy cascade'
    save(REPORTS/'MAIN_RESULTS.json',dict(status='RUNNING_SNAPSHOT',rows=rows,raw_online_primary=True,
         all_completed_seed_metrics_are_genuine_six_camera_pooled=True,old_different_phase_scores_not_spliced=True,
         original_GMT_reference_frontend_difference_disclosed=True,new_JEV_value='NOT_ESTABLISHED',
         true_old_PhaseVI_binary_same_contract='NOT_RUN_DIFFERENT_FEATURE_AND_ACTION_INTERFACE',
         new_binary_is_exact_pair_threshold_reparameterization_not_independent_architecture=True))
    ablations={row:{name:arms.get(name,dict(status='NOT_RUN',summary=None)) for name in names} for row,names in ABLATIONS.items()}
    shared=REPORTS/'TRAINED_CONTRACT_CALIBRATION_SEED20261009.json'
    if shared.exists():ablations['A5']['functional_same_weights_recompute']=read(shared)
    save(REPORTS/'ABLATIONS.json',dict(status='RUNNING_SNAPSHOT',values=ablations,
         evidence_removal_is_frozen_weight_inference_intervention=True,evidence_removal_proposals_can_change=True,
         current_identity_continuity_not_future_safety=True))
    parameters=read(REPORTS/'PARAMETERS_AND_FPS.json')
    new=REPORTS/'NEW_MODULE_PARAMETER_READS.json'
    if new.exists():parameters['new_JEVTrack']=read(new)
    fps={}
    for path in REPORTS.glob('EFFICIENCY_*_formal_SEED20261009_VIDEO17.json'):
        item=read(path);fps[item['arm']]=item
    required_fps=['parent','cosine','mlp','attention','jev','deepsets','fixed_question','no_shared_recompute','XIII_fixed_reference','XV_V3_reference','original_GMT']
    components={read(p)['arm']:read(p) for p in REPORTS.glob('COMPONENT_PROFILE_*.json')}
    parameters.update(status='COMPLETE' if all(n in fps for n in required_fps) and len(components)==5 else 'COUNTS_COMPLETE_FPS_OR_COMPONENTS_PENDING',
                      FPS=dict(status='COMPLETE' if all(n in fps for n in required_fps) else 'PARTIAL_OR_NOT_RUN',
                               cases=fps,missing_methods=[n for n in required_fps if n not in fps]),
                      instrumented_components=components,
                      new_parameters_reference='NEW_MODULE_PARAMETER_READS.json')
    save(REPORTS/'PARAMETERS_AND_FPS.json',parameters)
    matrix=REPORTS/'FORMAL_STRICT_IDENTITY_MATRIX.json';identity={}
    if matrix.exists():
        cases=read(matrix)['cases']
        for arm in sorted({c['arm'] for c in cases}):
            chosen=[c for c in cases if c['arm']==arm];counts=collections.Counter();comparison=collections.Counter()
            for case in chosen:
                counts.update(case['analysis']['counts']);comparison.update(case.get('independent_cold_run_comparison',{}))
            tp=counts['cross_known_correct_returns'];fp=counts['cross_known_wrong_returns'];rel=counts['cross_known_relevant_prior_observations']
            anchor=counts['known_rows_with_first_owner_correct_ID_anchor']
            identity[arm]=dict(completed_scene_seed_cases=len(chosen),keys=[[c['seed'],c['video']] for c in chosen],counts=dict(counts),
                independent_cold_run_comparison=dict(comparison),past_cross_ID_index_precision=tp/(tp+fp) if tp+fp else None,
                past_cross_ID_index_recall=tp/rel if rel else None,
                first_owner_correct_ID_preservation=counts['first_owner_correct_ID_preserved_rows']/anchor if anchor else None,
                observed_error_duration_camera_frames=sum(c['analysis']['observed_error_duration_camera_frames'] for c in chosen),
                native_IDs_with_multiple_known_GT=sum(c['analysis']['native_IDs_with_multiple_observed_GT'] for c in chosen),
                scene_details=[dict(seed=c['seed'],video=c['video'],episode_count=c['analysis']['observed_error_episode_count'],
                    duration_p50_p95=c['analysis']['observed_error_duration_p50_p95_camera_frames'],
                    right_censored=c['analysis']['right_censored_episodes'],
                    local_evidence=c['analysis']['detailed_local_analysis']) for c in chosen])
    save(REPORTS/'CROSS_CAMERA_ANALYSIS.json',dict(status='PARTIAL_SNAPSHOT' if identity else 'NOT_RUN',methods=identity,
         primary_retrieval='Actual opaque predicted Global ID lookup into already committed other-camera observations; no GT-selected whole-video mapping',
         pooled_pair_weighted_not_independent_observations=True,
         N01_N10_scope='Independent complete cold-start runs, own first-known owners; not equal-state causal effects',
         wrong_owner_time_is_observed_contiguous_camera_frames_unknown_gaps_right_censored=True,
         originals_not_proven_restored_by_IDSW_or_N01=True,duplicates_UNKNOWN=True))
    jev=arms['jev'];ordinary={n:v for n,v in arms.items() if n in ['mlp','deepsets','attention','linear','threshold','state_threshold'] and v['status']=='COMPLETE_THREE_SEEDS'}
    strongest=max(ordinary,key=lambda n:ordinary[n]['summary']['HOTA']['mean']) if ordinary else None
    delta={key:jev['summary'][key]['mean']-ordinary[strongest]['summary'][key]['mean'] for key in METRICS} if strongest and jev['status']=='COMPLETE_THREE_SEEDS' else None
    engineering=read(REPORTS/'MODEL_AND_NATIVE_CONTRACT_TESTS.json')['status']=='PASS' and shared.exists() and new.exists()
    save(REPORTS/'GO_NO_GO.json',dict(status='ACTIVE_FINITE_RESEARCH_AUDIT',original_MASTER_goal_achieved=False,
         engineering=dict(status='MODEL_CONTRACT_PASS_FULL_TIMING_PENDING' if engineering and parameters['status']!='COMPLETE' else 'PASS' if engineering else 'PARTIAL'),
         association=dict(status='NOT_SUPERIOR_TO_STRONGEST_COMPLETE_ORDINARY_CONTROL' if delta and delta['HOTA']<.5 else 'NOT_ESTABLISHED',
                          strongest_completed_ordinary_control=strongest,paired_mean_differences=delta),
         JEV_independent_value='NOT_ESTABLISHED',paper_readiness='NOT_READY',
         missing_efficiency_methods=parameters['FPS']['missing_methods'],algorithm_major_revision=1,
         significance_not_claimed_from_three_correlated_development_sequences=True,
         publication_only_negative_hypothesis_not_theoretical_impossibility=True,
         original_MASTER_long_term_safety_gate_unchanged=True))
    table=['| 方法 | 完成种子 | HOTA ± seed SD | AssA | CVIDF1 | CVMA | IDSW |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,item in arms.items():
        s=item['summary']
        if not s:continue
        sd=s['HOTA']['std_seed'];h='%.3f ± %.3f'%(s['HOTA']['mean'],sd) if sd is not None else '%.3f（单种子）'%s['HOTA']['mean']
        table.append('| %s | %d | %s | %.3f | %.3f | %.3f | %.1f |'%(arm,len(item['cases']),h,s['AssA']['mean'],s['CVIDF1']['mean'],s['CVMA']['mean'],s['IDSW']['mean']))
    q3='新JEV与最强已完成普通控制 %s 的 HOTA 均值差 %.3f，尚不支持独立架构优势。'%(strongest,delta['HOTA']) if delta else '普通控制矩阵仍未完成。'
    questions=[
      ('Q1：是否只是二次关联？','重复相同输入 Hungarian 必须保持分配；规则屏蔽边后重解是不同干预。真实评分×求解对照运行中。固定同状态 H64：parent181、JEV173、MLP205、Attention173 个已知错误观测；JEV与Attention相同。精确触发预算诊断单独保存，不替代完整在线结果。','PAIRED_NATIVE_RESULTS.json / SAME_STATE_BUDGET_DIAGNOSTIC.json'),
      ('Q2：阈值为什么不够？','固定、state/pair可学习门限正在同条件比较。新Binary与pair门限是同权重精确重参数化，不能冒充旧Phase VI独立Binary。严格同接口旧VI模型尚未重放。没有证据证明所有阈值都不够。','MAIN_RESULTS.json / MECHANISM_jev_pilot_SEED20261009.json'),
      ('Q3：MLP或普通Attention为什么不够？',q3+' 各新增控制共享相同冻结XIV初始化、证据、当前标签、最多4方案和4000更新。新模块JEV169092、MLP154980、Attention168964参数；不是整个历史预训练架构的独立公平比较。','FORMAL_CURRENT_RESULTS.json / NEW_MODULE_PARAMETER_READS.json'),
      ('Q4：参数和延迟？','完整原GMT53515309参数；新JEV36165558，总变化−17349751，但完整前端不同，不能把该差值解释为纯关联模块开销。共同前端parent35996466，新JEV增169092。完整视频3次重启FPS和单独组件profile仍以实际完成状态为准；MAC/FLOPs含自定义核缺口，明确PARTIAL。','PARAMETERS_AND_FPS.json'),
      ('Q5：是否真是JEV？','每摄像头因果关联事件共享State编码1次，实际重复编码2次同输出；动态合法ID/私有DEFER和问题条件化均实现。TRAIN16共同可认证两问题标签分歧0，不能据两个Head或embedding敏感性证明独立语义价值。目前应称JEV-inspired关联研究实现。','TRAINED_CONTRACT_CALIBRATION_SEED20261009.json / QUESTION_SEPARATION_AND_PLAN_COVERAGE.json')]
    (DOCS/'REVIEWER_FIVE_QUESTIONS.md').write_text('# 五个审稿人问题：当前可核查答案\n\n工作仍在执行，未完成项不视为通过。\n\n'+'\n\n'.join('**'+title+'**\n\n'+answer+'\n\n证据：`reports/JEV_ONEWEEK/'+evidence+'`。' for title,answer,evidence in questions)+'\n')
    twelve=[
      '原始完整GMT冻结参考HOTA78.110；前端1152D/Stage2权重不同。共同前端XIV parent三种子HOTA73.505、AssA73.639、CVIDF1 88.546；新版和旧参考见主表，不能跨训练阶段拼最佳值。',
      '新JEV在已完成三种子上未全面提高Global ID一致性：检索精确率、召回、原始ID保持和错误传播必须共同判断。',
      'N01记录实际同检测行从owner错误变为正确，但完整独立轨迹的ID namespace不同，不证明原始ID被恢复；详细与同状态H64结果分开。',
      'N10真实反改错完整保留，不能只报告IDSW下降。见CROSS_CAMERA_ANALYSIS里的各方法计数、UNKNOWN与右删失。',
      questions[0][1],questions[1][1],questions[2][1],
      '共享状态是一项真实计算复用，不是已证实的准确率创新。两问题在共同已知TRAIN16目标完全一致，独立语义增益尚未成立。',
      questions[4][1],questions[3][1],
      '新JEV主实验已为真实完全mutated-state online，所有检测/ID提交和Gallery/Bank会影响下一帧；主要HOTA/AssA相对MLP未达到预注册探索目标，CVIDF1优势不一致。',
      '目前不足以支撑声称JEV具有明确独立贡献的WWW论文。持续更新多摄像头身份索引与错误检索属于真实Web相关问题；研究动机不代替独立性能与可靠性证据。']
    report='# JEVTrack 本周实验报告（随真实完成工件更新）\n\n当前结论：JEV独立贡献 **NOT_ESTABLISHED**。此报告仍是进行中的证据汇总；正式对照、归因、消融及完整视频效率未全部完成前不宣布任务完成。\n\n'+'\n'.join(table)+'\n\n'
    report+='上述学习模型首先对DEV17/18/19六个相机进行真实pooled评测，再计算三个固定种子的均值/样本标准差；种子波动不是三个独立数据集或显著性证明。DetA/IDF1/MOTA/Frag和逐种子数据见MAIN_RESULTS。全部主指标为未做未来短轨筛选的完整在线结果，CVIDF1/CVMA来自冻结官方MATLAB。\n\n'
    for index,answer in enumerate(twelve,1):
        report+='**%d. %s**\n\n'%(index,answer)
    report+='复现代码、协议、START和权重SHA均绑定每个工件。模型保存在 `/data1/liuyeqiang/WWW_jev_oneweek_runtime/20261012_v1/training_v1/`，真实轨迹/详细事件留服务器；GitHub只传源代码、小指标、失败、配置与SHA清单。参数完整前向读数仅覆盖真实TRAIN12首2帧，未观察Bank分支不能当成永久Dormant。\n\n'
    report+='关键局限：新梯度只有TRAIN12/13/14非旧reserved时间块，TRAIN16与DEV均已旧暴露；前端旧Stage1训练曾见全部TRAIN。未读取20/21/22或官方TEST，不启动Full24，没有新下载。当前GT认证的困难选项多数UNKNOWN，置信度校准是预暴露控制器诊断而非独立泛化。完整原GMT前端和主表共同Stage1不同；M5是XIV/native规则，未冒充原完整GMT传统cascade；旧VI Binary接口不一致。原MASTER长期安全Goal及8TRAIN/3RES安全门均未通过、未完成。\n\n'
    report+='状态与原始工件：`EXPERIMENT_REGISTRY.json`、`MAIN_RESULTS.json`、`ABLATIONS.json`、`PARAMETERS_AND_FPS.json`、`CROSS_CAMERA_ANALYSIS.json`、`GO_NO_GO.json`、`ENGINEERING_FAILURES.json`、`ARTIFACT_SHA_MANIFEST.json`。缺失实验均为NOT_RUN/PARTIAL，不以零代替。下一步只完成冻结有限矩阵，不通过扩大模型、反复调DEV或挑选种子制造成功。\n'
    (DOCS/'FINAL_REPORT.md').write_text(report)


if __name__=='__main__':refresh()
