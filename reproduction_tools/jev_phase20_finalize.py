"""Freeze representation and Web retrieval diagnostics without training-gate changes."""
from jev_phase20_common import *


def main():
 protect();anchor=read(REPORTS/'PAST_ANCHOR_EVIDENCE_AUDIT.json')
 retrieval=read(REPORTS/'NATIVE_ACTION_RETRIEVAL_VALUE.json')
 temporal=read(REPORTS/'NATIVE_RETRIEVAL_TEMPORAL_DECOMPOSITION.json')
 assert anchor['summary']['events']==50 and len(retrieval['pairs'])==len(temporal['pairs'])==50
 a=anchor['summary'];assert a['baseline_correct']==a['anchor_correct']==35
 assert a['transitions']['WRONG_to_CORRECT']==a['transitions']['CORRECT_to_WRONG']==2
 for p in retrieval['pairs']:
  for h,b in p['H']['BASE'].items():
   q=p['H']['PROBE'][h]
   for field in ['queries','known_current_queries','same_GT_past_opposite_camera_available','known_queries_with_GT_available']:
    assert b['counts'].get(field,0)==q['counts'].get(field,0)
 # Actual-prefix and feature-source invariants ran in the scoring/metric actors.
 manifest=read(PRIOR/'reports/JEV_PHASE19/ARTIFACT_SHA_MANIFEST.json')
 for name,digest in manifest['files'].items():assert sha(PRIOR/name)==digest,name
 prior_gate=read(PRIOR/'reports/JEV_PHASE19/LEARNING_ELIGIBILITY.json')
 assert not prior_gate['qualified'] and prior_gate['safe_TRAIN']==3 and prior_gate['safe_RESERVED']==1
 decision=dict(status='BOUNDED_DIAGNOSTICS_COMPLETE',master_goal_complete=False,
  engineering='PASS_ACTUAL_CAUSAL_REPLAY_AND_SOURCE_INTEGRITY',
  data_eligibility='NO_GO_PRIOR_INDEPENDENT_SUPERVISION_3TRAIN_1RESERVED',
  early_representation='LOCAL_COMPLEMENTARITY_WITH_ANTI_CORRECTIONS;NO_GO_UNCONDITIONAL_REPLACEMENT',
  retrieval_mechanism='SUPPORTED_FOR3_ACTIONS_IN2_PRIOR_DEPENDENCY_UNITS;NO_TRAINED_POLICY_CLAIM',
  scientific_success=False,deployment='NOT_EVALUATED',independent_value='NOT_ESTABLISHED',
  new_training='NOT_RUN',new_native_actions=0,new_policy_metrics=None,paper_readiness='NOT_READY',
  prior_gate_unchanged=ref(PRIOR/'reports/JEV_PHASE19/LEARNING_ELIGIBILITY.json'),
  next_required='Independent manually valid ownership/causal utility supervision and validation; first establish authorized source access and audit metadata. No dependent training before eligibility.',
  sealed20_21_22_accessed=False,official_TEST=False,Full24=False)
 save(REPORTS/'FINAL_GO_NO_GO.json',decision)
 save(REPORTS/'ENGINEERING_AND_DATA_AUDIT.json',dict(status='PASS_WITH_EXPLICIT_SCIENTIFIC_LIMITS',
  fixed_population=anchor['population'],loaded_feature_payloads=anchor['loaded_payloads'],
  past_only_temporal_assertions=True,prototype_GT_inputs=False,candidate_truncation=False,
  existing_native_branches=66,new_native_branches=0,paired_query_GT_denominators_equal=True,
  frozen_Phase19_publication_files_verified=len(manifest['files']),
  prior_outcomes_known_before_diagnostics=True,exploratory_only=True,
  prototype_provenance=anchor['prototype_provenance'],runtime_GiB=size(OUT)/2**30,
  added_runtime_cap_MiB=64,home_free_GiB=shutil.disk_usage(OUT).free/2**30))
 safe=[p for p in temporal['pairs'] if p['owner_safe_qualification']];table=[]
 for p in safe:
  f=p['first_payload'];first=f['PROBE']['counts'].get('wrong_returned',0)-f['BASE']['counts'].get('wrong_returned',0)
  table.append('| '+ ' | '.join([str(p['selection_index']),p['variant'],str(first),
   str(p['later_deltas']['64']['wrong_returned']),str(p['later_prediction_ID_rows_changed'])])+' |')
 text=f'''# Phase XX — 过去身份表征与原生检索污染诊断

本轮继续长期科研Goal，完成两个不依赖训练资格的诊断：固定50组事件的早期原始观测表征，以及既有66个真实原生分支的跨摄像头身份检索效用。未新增视频读取、视觉提取、native动作、训练或测试解封。Phase19原始结果及3TRAIN＋1预留安全监督Gate保持冻结。

## 早期证据能否补充当前表征

动机：原native历史使用最近观测、Gallery全均值、own-camera均值、other-camera均值；混杂历史可能压过早期证据。固定一个无需训练的诊断：候选最早3个真实提交ReID向量取均值并归一化，与当前ReID算余弦；不足3个或原型为零则保持原四槽余弦分数。不清理历史、不使用GT筛观测、不删除合法候选。

复用原事前50组（18错误、32正确控制），不按新分数选事件。相机当前所有行评分先于该相机任何提交；帧0采用源码规定的最大检测数相机初始化，之后camera0→camera1。218个实际cache payload逐项SHA验证；每个候选原型的来源event/ID/SHA均可追溯，当前/未来观测进入候选时会assert失败。第一组事件的bootstrap来自已经执行的source-defined原生初始化，没有凭GT创造身份。

| 点式排名指标 | 原四槽余弦 | first3＋原分数fallback |
|---|---:|---:|
| 正确owner候选 | 35/50 | 35/50 |
| 已知owner覆盖 | 48/50 | 47/50 |
| TRAIN错误组正确 | 2/12 | 3/12 |
| TRAIN正确控制保持正确 | 24/24 | 23/24 |
| 预留错误组正确 | 1/6 | 1/6 |
| 预留正确控制保持正确 | 8/8 | 8/8 |

共10次排名变化，wrong→correct2次、correct→wrong2次、wrong→UNKNOWN2次、UNKNOWN→wrong1次；其余变化仍为正确别名或错误owner。42/50事件所有合法候选均有3个过去观测，其余使用显式fallback。按整个原census的物理owner/近分数冲突连接，50个事件只有16个依赖组件，不当成50次独立试验。

结果有局部互补证据，却不支持无条件替换现有历史表征。更高的“已知条件正确率”也不能掩盖覆盖减少。所有比较是候选点式排名，不是容量受限Hungarian或实际controller表现。此前原生结果已知，所以这是探索性审计；没有显著性、泛化或信息不可能性结论。

例子：video03帧728，first3将correct-owner从第2名排到第1名；但Phase19中该对应真实动作仍有副作用。另一个video03帧899案例，原四槽余弦选对owner，first3反而排到第14名。候选正确性、证据可靠性和执行后的长期价值必须分开，不能将新排名正例直接当安全动作监督。

## 真实分支的跨摄像头检索价值

服务定义：当前检测的实际预测GlobalID，查询此前已经提交的另一摄像头观测；每个camera batch查询完再插入当前观测。GT只在选定返回结果后评价，不能过滤索引、候选或查询。索引保留raw观测，包括UNKNOWN。前缀来自完整原生commit-observer日志，分支来自真实mutated-state H64输出。帧0最大相机的初始化没有该observer记录，因此在BASE/ALT中等价排除，并明确限制为观测日志语料，不冒充完整生产索引。

所有16BASE＋50ALT均评价H16/32/64。每对分支的GT查询数、同人物过去可用结果分母完全一致，跨相机precision、覆盖、UNKNOWN、recall均从计数计算。不存在跨相关替代未来的pooled总分或逐视频均值冒充pooled指标。

50个ALT中，H64有23个提高已知返回precision、16个降低；24个减少错误返回、16个增加错误返回。原5个合格安全owner动作中，3个提高precision且减少错误返回，另外2个没有观测到跨相机收益。其中2个动作属于同一个video07依赖组，所以只有2个不同的已冻结依赖单位（video07与预留10），不扩张成3个独立安全监督。

进一步分解避免将当前一次大查询误当长期收益：

| Phase19组号 | ALT | 当前camera payload错误返回Δ | 后续H64错误返回Δ | 后续预测ID变化行数 |
|---|---|---:|---:|---:|
{chr(10).join(table)}

后续3个有收益动作的预测ID都与BASE完全相同，却分别减少63、64、64次错误返回：一次正确写入减少了后续另一摄像头查询中的历史污染。它们是重复服务请求返回的观测计数，不是63/64个纠错人物，也不是错误身份持续了63/64帧。两个video07动作相互排斥且相关，不能把两次64相加当一次策略收益。部分UNKNOWN暴露变化发生在当前payload，也完整单独记录。

这提供了一个具体的WWW机制证据：可靠当前身份提交的价值可以表现为之后的身份索引质量，而不仅是之后预测ID变化。但动作来自oracle诊断，没有证明任何JEV或普通网络能安全选择它们，更没有建立新策略全视频独立增益。原Gate不回头改，新的检索效用也不把NO_GO升级为训练PASS。

## 执行、成本与限制

早期表征actor固定dba2612，检索actor26bb4b2，时间分解actor3ea285b；精确源码SHA、协议版本和原始路径在RESULT。早期与检索协议在各自新分数计算前冻结；时间分解明确是在初次检索汇总已知后追加的解释性审计。单CPU、0GPU，实际运行约13.6秒＋5.0秒及时间分解，新增留存远小于64MiB。没有新环境或权重下载，没有普通工程失败被伪装成科研失败。

Phase19全部37个发布文件SHA保持一致。历史模型、GT、预测、因果分支和失败日志未改。已验证配对查询分母、实际分支ID/GT和原型时间顺序；真实动态提交证据仍是Phase19的66分支，而不是本轮新执行66次。

## 下一步和主Goal

无条件first3替换不进入native策略升级；局部互补信息保留为未来可校准的候选证据。下一步先建立独立人工有效的身份与长期动作监督，以及可信验证关系，研究如何判断何时使用新证据。已完成[新数据范围方案](PHASE20_NEXT_DATA_SCOPE_PROPOSAL.md)：先仅审计来源/标注元信息≤64MiB，不下载视频、不训练。MMPTRACK如已有合法副本更适合先审计；公开WILDTRACK/SCOUT各有时间密度与人工GT范围限制，不能直接承诺补齐独立监督。

MASTER科学目标仍未完成：正式学习、同证据强对照、独立完整新策略验证、可靠REACTIVATION、统一生命周期与部署均尚缺资格或证据。当前论文NOT_READY。不得用探索性点式排名、局部oracle检索收益或旧DEV数字宣称JEV架构独立优越。

复核入口：PAST_ANCHOR_EVIDENCE_AUDIT、NATIVE_ACTION_RETRIEVAL_VALUE、NATIVE_RETRIEVAL_TEMPORAL_DECOMPOSITION、FINAL_GO_NO_GO、ARTIFACT_SHA_MANIFEST。所有原始大文件留服务器，仅代码、精简JSON、协议、报告和SHA清单推GitHub。
'''
 (ROOT/'docs/PHASE20_FINAL_RESEARCH_REPORT.md').write_text(text)
 goal=read(MASTER/'MASTER_GOAL.json')
 goal.update(current_phase_status='READONLY_EVIDENCE_COMPLETE_DATA_ACCESS_REQUIRED',master_status='PIVOT_REQUIRED',
  active_hypotheses=[],next_experiment='NEW_DATA_SOURCE_QUALIFICATION',scientific_SUCCESS=False,paper_readiness='NOT_READY')
 goal.setdefault('rejected_hypotheses',[]).append(dict(id='E20_UNCONDITIONAL_FIRST3',
  result='35/50same;2pointwisecorrections+2anticorrections,UNKNOWNcoverage loss; no native replacement or training'))
 goal.setdefault('completed_supported_hypotheses',[]).append(dict(id='E20_NATIVE_INDEX_POLLUTION',
  result='3 existing qualified actions in2dependencyunits reduce laterwrongreturns63/64/64 with0laterIDchanges; diagnostic mechanism only'))
 save(MASTER/'MASTER_GOAL.json',goal)
 registry=read(MASTER/'EXPERIMENT_REGISTRY.json')
 for e in registry['experiments']:
  e.update(status='COMPLETE',evidence=[ref(REPORTS/('PAST_ANCHOR_EVIDENCE_AUDIT.json' if e['id']=='E20_ANCHOR_AUDIT' else 'NATIVE_ACTION_RETRIEVAL_VALUE.json'))])
 registry['experiments'].append(dict(id='E20_RETRIEVAL_DECOMPOSE',status='COMPLETE',requires=['E20_RETRIEVAL_VALUE'],requires_learning_gate=False,
  description='Separate currentquery from laterindexpollution, actualID/GT paired',commands=[],evidence=[ref(REPORTS/'NATIVE_RETRIEVAL_TEMPORAL_DECOMPOSITION.json')]))
 save(MASTER/'EXPERIMENT_REGISTRY.json',registry)
 save(MASTER/'NEXT_ACTION.json',dict(status='PIVOT_REQUIRED_DATA_ACCESS_OR_SCOPE',scientific_success=False,
  next='Establish authorized independent data source and run bounded annotation/split metadata audit.',
  proposal=str(ROOT/'docs/PHASE20_NEXT_DATA_SCOPE_PROPOSAL.md'),no_training_or_image_downloads=True,
  blocker='Current native safe supervision3TRAIN/1RESERVED, no established authorized independent manually-valid data source.'))
 with (MASTER/'RESEARCH_DECISIONS.md').open('a') as f:
  f.write('\n\nPhase XX: fixed50 past-onlyfirst3 evidence audit:35correct vs35baseline,2corrections/2anti-corrections; no unconditional replacement. Existing66nativebranches show3safeactions in2units reduce laterindexwrongreturns63/64/64 with unchanged laterIDs. Exploratory mechanisms, no newpolicy claim or gate change. Independent source/annotation qualification proposed before further learning.\n')
 print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
