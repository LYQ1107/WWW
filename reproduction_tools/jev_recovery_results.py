"""Real pooled metrics, six independent gate axes and explicit unfinished deliverables."""
import collections,numpy as np
from jev_recovery_common import *
def refresh():
 from run_early_pilot_tracking import extract_metrics
 arms=collections.defaultdict(list);cases=[]
 for file in REPORTS.glob('R*/MATLAB_*_formal_SEED*_VIDEOS_17_18_19.json'):
  result=read(file);prefix=file.parent.name;arm=result['arm'];seed=result['seed']
  folder=OUT/prefix/'matlab_v1/formal'/arm/f'seed{seed}/17_18_19'
  evalroot=folder/'pooled_TrackEval/tracking_eval_runtime_state/native/evaluation'
  metric=extract_metrics(evalroot);metric.update(CVIDF1=result['CVIDF1'],CVMA=result['CVMA'])
  row=dict(arm=arm,seed=seed,track=prefix,metrics=metric,official=ref(file),TrackEval=ref(evalroot/'metrics.json'));cases.append(row);arms[prefix,arm].append(row)
 summary=[]
 for (track,arm),rows in sorted(arms.items()):
  summary.append(dict(track=track,arm=arm,status='COMPLETE_THREE_SEEDS' if len(rows)==3 else 'PARTIAL',completed_seeds=[r['seed'] for r in rows],metrics={k:dict(mean=float(np.mean([r['metrics'][k] for r in rows])),sample_seed_SD=float(np.std([r['metrics'][k] for r in rows],ddof=1)) if len(rows)>1 else None) for k in ['HOTA','DetA','AssA','IDF1','MOTA','IDSW','Frag','CVIDF1','CVMA']}))
 registry_path=REPORTS/'EXPERIMENT_REGISTRY.json'
 registry=read(registry_path) if registry_path.is_file() else {}
 registry_jobs=registry.get('jobs',{}) if isinstance(registry,dict) else {}
 reviewed_failed_jobs=sorted(name for name,value in registry_jobs.items() if isinstance(value,dict) and value.get('status')=='FAILED')
 formal_routes_complete=bool(summary) and all(row.get('status')=='COMPLETE_THREE_SEEDS' for row in summary)
 queue_audit_path=REPORTS/'R2_R3_QUEUE_TERMINAL_AUDIT_20261014.json'
 queue_audit=read(queue_audit_path) if queue_audit_path.is_file() else {}
 queue_terminal=queue_audit.get('queue_terminal') is True
 online_status='COMPLETE_BOUNDED_FORMAL_RESULTS_WITH_REVIEWED_QUEUE_FAILURES' if formal_routes_complete and queue_terminal and reviewed_failed_jobs else ('COMPLETE_BOUNDED_FORMAL_RESULTS' if formal_routes_complete else 'PARTIAL_UNTIL_FROZEN_JOBS_COMPLETE')
 save(REPORTS/'FULL_ONLINE_RESULTS.json',dict(status=online_status,cases=cases,summary=summary,formal_routes_complete=formal_routes_complete,queue_terminal=queue_terminal,reviewed_failed_jobs=reviewed_failed_jobs,scientific_success_inferred=False,no_missing_metrics_imputed=True,pooled_six_camera_first=True,three_seeds_not_significance=True,pre_exposed_DEV_not_independent=True))
 save(REPORTS/'FAIR_BASELINE_RESULTS.json',dict(status='ACTUAL_COMPLETED_CASES_ONLY',cases=[c for c in cases if c['arm'] in ['parent','threshold','mlp','attention','mlp_fusion','attention_fusion','f0_mean']],same_evidence_and_updates=True))
 training=[]
 for p in REPORTS.glob('TRAIN_R*.json'):
  value=read(p)
  training.append(dict(source=ref(p),**{k:value.get(k) for k in ['arm','phase','seed','revision','base','total_updates','maximum_plans','engineering_training_pass','rank_supervised_batches','rank_nonzero_batch_fraction','certified_changed_row_groups_with_nonzero_ranking_head_gradient','checkpoint']}))
 save(REPORTS/'TRAINING_RESULTS.json',dict(status='COMPLETED_SNAPSHOT',cases=training,full_histories_in_individual_reports=True))
 inherited=ref(ROOT/'reports/JEV_ONEWEEK/FORMAL_CURRENT_RESULTS.json')
 gates=[read(p) for p in REPORTS.glob('R*_MECHANISM_GATE.json')]
 goal=read(REPORTS/'MASTER_GOAL.json');goal.update(status='ACTIVE_RESEARCH',revision_gates=gates,scientific_success=False,JEV_INDEPENDENT_VALUE='NOT_ESTABLISHED',CROSS_CAMERA='BOUNDED_R2_R3_FORMAL_COMPLETE_JEV_INDEPENDENT_NOT_ESTABLISHED',EFFICIENCY='BOUNDED_PRIMARY_TIMINGS_COMPLETE_MASTER_DEPLOYMENT_NOT_ESTABLISHED',independent_validation='INELIGIBLE on old DEV');save(REPORTS/'MASTER_GOAL.json',goal)
 save(REPORTS/'GO_NO_GO.json',dict(status='ACTIVE_BOUNDED_RESEARCH',MASTER_COMPLETE=False,JEV_INDEPENDENT_VALUE='NOT_ESTABLISHED',completed_route_gates=gates,new_scientific_success=False,inherited_frozen_reference=inherited,strict_success_rule=read(REPORTS/'PREREGISTRATION.json')['success'],independent_validation=False,rule='No claim of JEV architecture value from structural components, coverage gain, or partial/missing seeds'))
 lines=['# JEVTrack 持续研究报告','','MASTER 仍在执行。下表只读取实际已完成官方合并评价，不用旧结果填新方法数字。已有 DEV 不属于独立验证。','','| Track | 方法 | 种子数 | HOTA | AssA | CVIDF1 | IDSW |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
 for s in summary:
  m=s['metrics'];lines.append(f"| {s['track']} | {s['arm']} | {len(s['completed_seeds'])}/3 | {m['HOTA']['mean']:.3f} | {m['AssA']['mean']:.3f} | {m['CVIDF1']['mean']:.3f} | {m['IDSW']['mean']:.1f} |")
 lines+=['','未完成的同状态干预、强基座正式实验、身份回归及完整 FPS 均仍是验收项。不能以脚本完成或单个负结果结束 MASTER。详见 EXPERIMENT_REGISTRY.json、NEXT_ACTION.json 及本地 PROGRESS。']
 (DOCS/'FINAL_RESEARCH_REPORT.md').write_text('\n'.join(lines)+'\n')
 (DOCS/'PHASE13_STRONG_BASE_EXPERIMENT.md').write_text('# 强 Fixed 基座\n\nXIII Fixed 原20k权重只读，新增残差零初始化。真实 TRAIN12 96场景原生提交、状态和预测完全等价后才允许训练。强基座阈值、MLP、Attention 与 JEV 使用同一 Base、数据、候选、标签和4000累计更新。R3 根据 TRAIN 机制门槛只迁移合格部分；旧75.637仅是历史系统参考，不能替代本轮公平实验。\n')
 (DOCS/'REVIEWER_FIVE_QUESTIONS.md').write_text('# 五个审稿人问题\n\nQ1：旧同分数重复 Hungarian 是无变化；新候选生成需单独归因，所有控制获得同一方案。新评分器×求解器分离仍待执行。\n\nQ2：Pair Threshold 是强控制；同合法池及同预算直接比较，不以动态风险口号替代具体身份增量。\n\nQ3：纯 MLP、标准 Attention 与 F1 共享问题 Reader 后的 MLP 融合分别列出，F1只检验融合。相同 TRAIN 批次 SHA 要核验。\n\nQ4：旧3.703FPS只是旧模型参考；新真实图像全场景三次重启、活跃参数和Stage2时间尚待实际测量，不填写新数字。\n\nQ5：共享状态、问题条件读取和合法联合方案是结构功能；是否独立优于普通方法须由新控制和消融证明，目前 NOT_ESTABLISHED。原始独立监督 Gate 保持不变。\n')
 print('RECOVERY_RESULTS_REFRESH',len(cases),flush=True)
if __name__=='__main__':refresh()
