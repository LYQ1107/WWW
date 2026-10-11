"""Publish bounded evidence and honest gates, without unlocking training or TEST."""
from jev_phase19_common import *


def main():
    protect()
    paired=read(REPORTS/'NATIVE_PAIRED_OWNER_OUTCOMES.json');s=paired['summary']
    assert s['total_complete_native_branches']==66 and s['strict_CLEAR_unevaluable_action_variants']==0
    assert s['learning_gate']=='NO_GO_INDEPENDENCE_OR_SAFETY'
    oldroot=Path('/home/liuyeqiang/WWW_jev_phase18')
    prior=read(oldroot/'reports/JEV_PHASE18/ARTIFACT_SHA_MANIFEST.json')
    for name,digest in prior['files'].items():assert sha(oldroot/name)==digest, name
    checkpoint=Path('/home/liuyeqiang/WWW_jev_phase15_runtime/20261010_v1/training_full_payload_v3/F_full/seed20261009/pilot/LAST_FROZEN.pth')
    assert sha(checkpoint)==read(REPORTS/'PREREGISTRATION.json')['frozen_checkpoint_SHA256']
    assert sha(STAGE1)==STAGE1_SHA
    save(REPORTS/'FROZEN_PRIOR_INTEGRITY.json',dict(status='PASS',publication_files=len(prior['files']),
        source_manifest=ref(oldroot/'reports/JEV_PHASE18/ARTIFACT_SHA_MANIFEST.json'),
        old_results_not_overwritten=True,checkpoint=ref(checkpoint),stage1=ref(STAGE1)))
    metrics={};latencies={};cache_sources=[]
    census=read(REPORTS/'COMPLETE_NATIVE_OWNER_CENSUS.json')
    table=[]
    for v in SCOPE:
        p=census_case(v)/'RESULT.json';r=read(p);x=r['strict_evaluation']
        assert not x['permissive_override']
        metrics[str(v)]=dict(split='TRAIN' if v in TRAIN else 'RESERVED',frames=r['frames'],
            source=ref(p),status=x['status'],metrics=x['metrics'],prepared=x['prepared'],
            raw_metrics=x.get('raw_metrics'),raw_GT_unchanged=True)
        c=OUT/'stage1_cache_v1'/f'video{v:02d}/RESULT.json';cr=read(c)
        assert cr['status']=='COMPLETE' and cr['Stage1_SHA256']==STAGE1_SHA
        cache_sources.append(ref(c));latencies[str(v)]=cr['latency_camera_payload_ms']
        counts=census['per_video_counts'][str(v)];m=x['metrics']
        vals=[f'{m[k]:.3f}' for k in ['HOTA','AssA','IDF1','MOTA']] if m else ['null']*4
        table.append('| '+ ' | '.join([f'{v:02d}',metrics[str(v)]['split'],str(r['frames']),
            str(counts.get('AVAILABLE_ERROR',0)),str(counts.get('UNAVAILABLE_ERROR',0)),*vals,
            str(int(m['IDSW'])) if m else 'null'])+' |')
    save(REPORTS/'FROZEN_V3_FULL_VIDEO_BASELINES.json',dict(status='COMPLETE_FROZEN_REFERENCE_ONLY',
        per_video=metrics,new_policy_comparison=None,full_video_strict=len([x for x in metrics.values() if x['metrics']]),
        annotation_affected=[v for v,x in metrics.items() if x['metrics'] is None],
        aggregation=None,reason='No mixed duplicate-GT aggregate or new-policy delta; all runs use frozen XV V3.'))
    eligibility=dict(status='NO_GO_INDEPENDENCE_OR_SAFETY',qualified=False,
        safe_TRAIN=s['independent_safe_TRAIN_groups'],safe_RESERVED=s['independent_safe_RESERVED_groups'],
        required_TRAIN=8,required_RESERVED=3,source=ref(REPORTS/'NATIVE_PAIRED_OWNER_OUTCOMES.json'),
        anti_correction_controls_executed=True,new_training='NOT_RUN',training_updates=0,
        fair_new_learned_controls='NOT_RUN',full_cold_online_new_policy='NOT_RUN',
        significance='NOT_TESTED',unseen_validation='NOT_ESTABLISHED',
        prospective_sampler_ceiling=dict(TRAIN=7,RESERVED=3,
            timing='whole-census graph frozen before intervention outcomes',
            interpretation='Current sample cannot reach8 TRAIN even if every component were safe; insufficient bounded dataset design/coverage, not model training failure.'),
        gate_not_relaxed_after_results=True)
    save(REPORTS/'LEARNING_ELIGIBILITY.json',eligibility)
    usage=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
    resources=dict(status='WITHIN_AUTHORIZED_CAP',runtime_GiB=usage/2**30,runtime_cap_GiB=8,
        home_free_GiB=shutil.disk_usage(OUT).free/2**30,home_reserve_GiB=30,
        cache_sources=cache_sources,cache_camera_payloads=sum(read(x['path'])['payloads'] for x in cache_sources),
        per_video_camera_payload_latency_ms=latencies,workers_max=6,
        intervention_queue=ref(OUT/'intervention_queue_v1/RESULT.json'),
        native_future_queue_wall_seconds=read(OUT/'intervention_queue_v1/RESULT.json')['seconds'],
        real_image_full_online_new_policy_FPS=None,deployment_latency_qualified=False,
        latency_scope='Actual frozen Stage1 camera-payload extraction, plus bounded cached-native futures queue wall time; not full online scene FPS.')
    assert resources['runtime_GiB']<=8 and resources['home_free_GiB']>=30
    save(REPORTS/'LATENCY_AND_RESOURCE_ACCOUNTING.json',resources)
    decision=dict(status='BOUNDED_PHASE_COMPLETE_PIVOT_REQUIRED',master_goal_status='ACTIVE',
        scientific_success=False,paper_readiness='NOT_READY',engineering='PASS_REAL_NATIVE_FORK_PARITY',
        complete_native_branches=66,verified_original_replay_payloads=13497,
        native_solver='original constrained remaining-row Hungarian, full mutated commit/future; no second MATCH scoring round',
        learning=eligibility['status'],new_training='NOT_RUN',new_policy_metrics=None,
        independent_value='NOT_ESTABLISHED',deployment='NOT_EVALUATED',
        data_design='INSUFFICIENT_QUALIFIED_INDEPENDENT_SUPERVISION',
        evidence=ref(REPORTS/'NATIVE_PAIRED_OWNER_OUTCOMES.json'),
        next_research='Freeze this round. Review sampling ceiling, annotation validity and historical exposure before any future scope. Prioritize evidence/calibration and preservation of correct assignments; additional architecture or training not justified by this diagnostic.',
        no_reward_resampling=True,no_gate_relaxation=True,sealed20_21_22_accessed=False,official_TEST=False,Full24=False)
    save(REPORTS/'FINAL_GO_NO_GO.json',decision)
    text=f'''# Phase XIX — 扩展范围的原生身份纠错证据报告

用户已批准追加 TRAIN02–09、预留10/11/15 的数据审计与有界实验。本轮已完成11个视频的完整冻结策略轨迹、50组事前抽样和16组/66个真实原生分支。工程与原生配对验证通过；独立安全监督只有 TRAIN3组、预留2组，低于冻结的8＋3要求。结论是当前监督与验证资格不足：不进入训练，也不把未执行的训练解释为架构失败。WWW主目标仍 ACTIVE，独立增益未建立、论文 NOT_READY。

## 范围、暴露与标注

8＋3视频划分在收益检查前固定。20/21/22、官方TEST和Full24未访问。Stage1已有all24训练暴露；XV V3本身直接训练12/13/14/16并继承旧初始化，不能将新范围描述成整个系统从未见过的数据。02/03/05、04、06/07、09/10/11均有不同程度历史实验记录；08/15也没有足够依据宣称绝对未暴露。10/11/15只是本轮预留验证，不是新独立TEST。

原始GT未修改。02、03、15有重复身份标注，所以完整视频严格指标为null；仅将从帧0至H64未来结束都唯一的前缀纳入严格干预。所有66分支的H16/32/64均完整、可严格评估，无宽松重复GT覆盖。

## 完整冻结 V3 轨迹

下表是原有策略的完整轨迹参考值，不是新模型或新规则的收益。AVAILABLE/UNAVAILABLE以过去证据建立的owner估计与合法active MATCH为范围；UNAVAILABLE不证明Bank或全部历史中不存在正确身份。百分制指标保持原始评估定义，不混合GT无效视频做总分。

| video | 划分 | scene frames | AVAILABLE_ERROR | UNAVAILABLE_ERROR | HOTA | AssA | IDF1 | MOTA | IDSW |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

感知共{resources['cache_camera_payloads']:,}个camera payload。05/06没有AVAILABLE_ERROR；02的错误不满足唯一GT前缀及完整未来要求。抽样得到18组AVAILABLE_ERROR、32组CORRECT_CONTROL，总50组；未补造64组，也没有按后续收益更换序列。干预为12错误组＋4正确控制组。错误组中12/12正确owner在合法MATCH内，11/12在GT-free替代动作里；video15的一个候选覆盖缺口保留。该条件化样本的cosine top1仅2/12正确，不能外推总体错误率或证明视觉信息不可能识别身份。

## 真实原生干预结果

16 BASE＋50 ALT＝66分支，全都实际改变状态并执行H64未来。完整原始回放核验13,497个camera提交payload：WHO/REACT引用、分数、合法mask、提交ID、视觉向量hash及raw-memory hash逐项相同。分支起点包括Instances/Gallery/hits/Bank/Commitment/RNG；每个BASE的未来提交与原始轨迹相同。替代动作从冻结GT-free候选生成，GT只用于离线标签与观察，不参与评分、mask或动作选择。约束一次目标行后，其余行使用原始Hungarian求解器和合法容量；该native实现不存在第二轮MATCH评分验收。

| 计数 | 结果 |
|---|---:|
| ALT动作 | 50 |
| 有真实纠错 | 19 |
| 有副作用 | 43 |
| 同时纠错与副作用 | 13 |
| 所有H均安全的纠错动作 | 6 |
| 安全时间代表 | 5 |
| 合并全轨迹依赖后的安全TRAIN组 | 3 / 8所需 |
| 合并全轨迹依赖后的安全预留组 | 2 / 3所需 |
| 严格CLEAR不可评估动作 | 0 |

19与43是相交集合，不应相减当净收益；多动作与多H也不是独立试验。真实纠错仅含known-wrong→known-correct或有过去owner支持的目标false-split修复；wrong→UNKNOWN不算纠错。任何H发生反改错、正确覆盖变UNKNOWN、新混入/false-split、错误观测增加、任一GT-camera最长观测错误段延长或严格CLEAR IDSW增加，都算副作用。观察缺口和窗口结束作截尾，没有伪造不可见时间的传播长度。

安全动作位于TRAIN03、04、07以及预留10、11。4个正确控制组的11个替代动作全部存在副作用，显示必须保留正确关联。部分纠错会通过全局改配或后续Bank路径发生，不能只按目标行即时标签建训练正例。每个动作的H16/32/64转移、污染、传播段、IDSW以及其他原MATCH行改配均在NATIVE_PAIRED_OWNER_OUTCOMES.json。

## 独立性与学习门槛

原selected-event依赖审计保留为诊断；在首个干预结果之前，额外冻结整段census的owner/近分数冲突图，包括未抽样事件的桥接连接。11个selected-only组件合并成10个整轨迹组件，跨时间03/04/07冲突不能重复计数；没有TRAIN/RESERVED混合组件。复算全图与冻结组件逐项一致，但这仍不是统计独立性或人物全新性的证明。

应明确承认设计限制：本轮固定代表在全图审计后最多只有7个TRAIN组件，即使全部安全也达不到8个的门槛。这个上限在干预结果前已冻结。本轮继续完成因果诊断，但不降低门槛、不追加有利代表来凑资格。因此NO_GO同时包含样本设计/覆盖不足和已观测的安全副作用，不能单凭这个门槛失败拒绝JEV架构。

新的controller训练、三类学习控制器对照、全视频新策略冷启动及部署测试均NOT_RUN；新策略HOTA/AssA/IDF1/MOTA增益为null。6个安全oracle动作说明有局部可纠错机会，并不证明某个可部署策略能够选到它们，更不证明JEV显著优于传统控制器。

## 工程修改、复核与成本

新增Phase19冻结感知提取、完整census、严格评估、原生fork、分层抽样、配对汇总及恢复控制脚本，继承原有native solver。首次census导出忘记接收forward返回值；10个已完成轨迹通过真实提交日志恢复导出，不重新生成身份。另一个旧cache writer将空(0,1024)改成(0,0)，导致video04帧627失败；reader仅恢复零行tensor的已知列维度，缓存原文件及非空值未改。只重跑04，出错前1,253提交和1,262评分查询完全一致。失败日志和原始版本都保留，1轮工程修复未涉及参数调优。

运行源码分别固定为cache fad03f8、census 07ac1d5、修复 aae6738、interventions d4ccc31；精确commit/SHA和原始路径保存在各RESULT与索引。工程检查来自真实11视频、66分支与hash验证，不以实现镜像单元测试替代。

最多6个GPU worker，优先无进程卡，允许显存足够时共用且不停止他人进程。66分支队列耗时{resources['native_future_queue_wall_seconds']:.1f}秒；这包含缓存native回放与分支，不是实时多相机吞吐。本轮留存约{resources['runtime_GiB']:.3f}GiB（上限8），home可用约{resources['home_free_GiB']:.2f}GiB（保底30）。实际Stage1 camera-payload p50/p95按视频记录，但不将其冒充完整在线scene FPS。

## 决策与交付

冻结当前证据，停止本轮MATCH扩展和训练。后续需要先解决标注有效性、历史暴露、独立样本覆盖及当前采样门槛的可达性，并事前重设合理实验人口与预算；优先研究可用证据/校准和正确关联的保持，而非凭局部oracle纠错宣称新架构有效。当前未批准新训练，不自动解封测试。

代码、精简JSON、报告和SHA清单推送GitHub；模型、视频、cache、完整journal及native状态只留服务器。复核入口为NATIVE_PAIRED_OWNER_OUTCOMES.json、FROZEN_V3_FULL_VIDEO_BASELINES.json、LEARNING_ELIGIBILITY.json、FINAL_GO_NO_GO.json及ARTIFACT_SHA_MANIFEST.json。主研究目标未完成，当前授权的扩展范围审计与有界实验已完成。
'''
    (ROOT/'docs/PHASE19_FINAL_RESEARCH_REPORT.md').write_text(text)
    goal=read(MASTER/'MASTER_GOAL.json');pre=read(REPORTS/'PREREGISTRATION.json')
    goal.update(current_phase_status='BOUNDED_EVIDENCE_COMPLETE_LEARNING_NO_GO',master_status='PIVOT_REQUIRED',
        active_hypotheses=[],remaining_authorized_budget=pre,next_experiment='REVIEW_EVIDENCE_AND_NEXT_SCOPE',
        scientific_SUCCESS=False,paper_readiness='NOT_READY')
    goal.setdefault('rejected_hypotheses',[]).append(dict(id='E19_H1',result='INSUFFICIENT_INDEPENDENT_SAFE_SUPERVISION;3TRAIN+2RESERVED; no new training, not architecture failure'))
    save(MASTER/'MASTER_GOAL.json',goal)
    registry=read(MASTER/'EXPERIMENT_REGISTRY.json')
    for e in registry['experiments']:
        if e['id']=='E19_NATIVE_FUTURES':
            e.update(status='COMPLETE',actor_source_commit='d4ccc31c1bb63665c9d35eb95f7bc9e1293dac78',
                description='16 fixed groups /66 real native branches, all complete',
                evidence=[ref(OUT/'intervention_queue_v1/RESULT.json'),ref(REPORTS/'NATIVE_PAIRED_OWNER_OUTCOMES.json')])
        if e['id']=='E19_QUALIFY':
            e.update(status='NO_GO',evidence=[ref(REPORTS/'LEARNING_ELIGIBILITY.json'),ref(REPORTS/'FINAL_GO_NO_GO.json')])
    save(MASTER/'EXPERIMENT_REGISTRY.json',registry)
    save(MASTER/'NEXT_ACTION.json',dict(status='PIVOT_REQUIRED',scientific_success=False,
        action=decision['next_research'],completed_authorized_round=True,new_training_authorized=False,
        reason='Qualified supervision3TRAIN/2RESERVED below8/3; design ceiling7TRAIN, historical validation exposure and anti-correction harms.'))
    with (MASTER/'RESEARCH_DECISIONS.md').open('a') as f:
        f.write('\n\nPhase XIX: approved02–09 + reserved10/11/15,11 complete frozen trajectories,66 native futures. 6safe actions in5 representatives =>3TRAIN/2RESERVED owner-conflict units;8/3 gate fails. Full-census graph ceiling7TRAIN disclosed. No training or architecture verdict. Bounded authorization complete; master science pending evidence/scope review.\n')
    print(json.dumps(dict(decision=decision,resources={k:resources[k] for k in ['runtime_GiB','home_free_GiB']}),indent=2))


if __name__=='__main__':main()
