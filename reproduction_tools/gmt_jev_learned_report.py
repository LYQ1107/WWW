"""Current learned-gate evidence, preserving historical negative decisions."""
from gmt_jev_common import *

KINDS=['mlp','attention','cic']

def main():
 zero=REPORTS/'LEARNED_GATE_ZERO_NATIVE_PARITY.json';pre=REPORTS/'LEARNED_GATE_SUPERVISION_PRECHECK.json'
 training=REPORTS/'LEARNED_GATE_SMOKE_TRAINING.json';progress=REPORTS/'LEARNED_GATE_TRAINING_PROGRESS.json'
 trained=read(training) if training.exists() else None
 revision_status=trained['status'] if trained else 'REAL_SUPERVISION_PRECHECK_PENDING_NO_UPDATES'
 updates=sum(case['steps'] for case in trained['cases'].values()) if trained else 0
 if progress.exists():
  live=read(progress);updates=max(updates,sum(c['steps'] for c in live['completed_heads'].values())+live['step'])
 previous=read(REPORTS/'TRAINING_METRICS.json')
 history=previous.get('historical_frozen_conflict_decision',previous)
 current=dict(status=revision_status,new_gradient_updates=updates,historical_frozen_conflict_decision=history,
  gate_revision_protocol=ref(DOCS/'LEARNED_GATE_SMOKE_PROTOCOL.md'),native_zero_precheck=ref(zero) if zero.exists() else None,
  supervision_precheck=ref(pre) if pre.exists() else None,training=ref(training) if training.exists() else None,
  fixed_endpoint_only=True,independent_validation=False,scientific_success=False)
 save(REPORTS/'TRAINING_METRICS.json',current)
 lines=['# 单次学习式风险门控修订结果','',
  '旧 conflict 门控、Rule 完整 TRAIN 负向结果及旧零训练决策仍保留。当前是由 TRAIN 认证目标覆盖缺口支持的一次有界修订，不能用 ORACLE 见证直接宣称新模型有效。','',
  f'当前阶段：{revision_status}；已记录 optimizer 更新总数：{updates}（各头单独计数）。',
  '三个头使用相同原生24维输入、候选/求解器、±0.05残差上界、原支持长度和提交规则。参数 MLP5119、Attention4898、CIC4962。',
  '32场景帧的三个原生零残差合同均通过；此检查不是完整视频或独立验证。']
 if pre.exists():
  p=read(pre);lines+=['',f"认证监督前置检查：{p['status']}；临时缓冲峰值 {p['buffer']['peak_accounted_buffer_bytes']/2**20:.3f} MiB。",
   'TRAIN与时间留出桶计数：', '', '```json',json.dumps(p['counts'],ensure_ascii=False,indent=2),'```',
   '时间留出桶的矩阵不参与梯度更新；GT只作离线认证，UNKNOWN、不唯一、阈值不可达及无联合有界纠错见证的错误行被mask。原生完整矩阵仍参与训练求解，UNKNOWN候选竞争不能忽略。',
   '时间桶并非严格标签隔离：原生轨迹与离线历史认证跨桶连续，较早留出桶的历史GT可参与随后TRAIN目标的历史认证。该桶仅用于诊断，不报告为独立或严格标签隔离验证；当前完整闭环仅在已暴露TRAIN16开发视频上评测。']
 if trained:
  lines+=['','| 头 | 固定终点 | 新增参数 | 训练样本顺序 SHA |','|---|---:|---:|---|']
  for kind,case in trained['cases'].items():
   assert sha(case['checkpoint']['path'])==case['checkpoint']['SHA256']
   lines.append(f"| {kind} | {case['steps']} | {case['parameters']} | {case['schedule_SHA256']} |")
  lines+=['','AdamW lr0.001、weight_decay0.0001、FP32、clip1、seed20261009；结构化 assignment hinge + threshold hinge +10×preserve +residual penalty。训练loss不证明完整闭环质量。']
 baseline=read(REPORTS/'ORIGINAL_GMT_METRICS.json');methods={}
 lines+=['','| 完整冷启动 TRAIN16 | HOTA | ΔHOTA | AssA | CVIDF1 | IDSW | Frag | 完整图像FPS |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for kind in KINDS:
  path=REPORTS/f'ONLINE_{kind}_learned.json'
  if not path.exists():lines.append(f'| {kind} | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |');continue
  result=read(path);assert result['status']=='COMPLETE' and result['raw_frontend_prediction_parity']
  from gmt_jev_learned_temporal import evaluate_temporal
  temporal=evaluate_temporal(kind)
  raw=result['primary_raw'];m=raw['metrics'];b=baseline['metrics'];speed_path=REPORTS/f'SPEED_{kind}_learned.json'
  delta={k:m[k]-b[k] for k in m};delta['CVIDF1']=raw['CVIDF1']-baseline['CVIDF1']
  gates=dict(HOTA=m['HOTA']>=b['HOTA']+.5,AssA=m['AssA']>=b['AssA'],CVIDF1=raw['CVIDF1']>=baseline['CVIDF1'],
   IDSW=m['IDSW']<=1.05*b['IDSW'],Frag=m['Frag']<=1.05*b['Frag'],
   descriptive_long_error=result['identity']['identity_inconsistency_duration_camera_frames']<=1.05*baseline['identity']['identity_inconsistency_duration_camera_frames'],
   genuinely_causal_error_propagation='NOT_ESTABLISHED')
  fps=None;ratio=None
  if speed_path.exists():
   s=read(speed_path);assert s['physical_GPU']==7 and len(s['trials'])==3
   original=read(REPORTS/'SPEED_original.json');assert s['GPU_UUID']==original['GPU_UUID']
   assert s['trained_checkpoint']['SHA256']==result['trained_checkpoint']['SHA256']
   fps=2*s['mean_full_scene_FPS'];ratio=s['mean_full_scene_FPS']/original['mean_full_scene_FPS']
  methods[kind]=dict(report=ref(path),primary_raw=raw,secondary_stock=result['secondary_stock'],delta=delta,
   quality_gates=gates,joint_measured_quality_pass=all(v for k,v in gates.items() if k!='genuinely_causal_error_propagation'),
   full_image_FPS=fps,FPS_ratio_to_GMT=ratio,efficiency_gate_pass=ratio>=.95 if ratio is not None else None,
   speed_report=ref(speed_path) if speed_path.exists() else None,paired_current_state_counts=result['paired_commit_counts'],
   correct_accepted_reference_preservation_rate=result['correct_accepted_reference_preservation_rate'],
   post_hoc_past_anchor_temporal=temporal,
   scientific_success=False,independent_validation=False)
  fps_text=f'{fps:.4f}' if fps is not None else 'PENDING'
  lines.append(f"| {kind} | {m['HOTA']:.6f} | {delta['HOTA']:+.6f} | {m['AssA']:.6f} | {raw['CVIDF1']:.6f} | {m['IDSW']:.0f} | {m['Frag']:.0f} | {fps_text} |")
 cache_path=REPORTS/'SPEED_mlp_learned_cached.json';cached=None
 if cache_path.exists():
  s=read(cache_path);p=read(REPORTS/'LEARNED_MLP_FEATURE_CACHE_PARITY.json');original=read(REPORTS/'SPEED_original.json')
  assert p['status']=='PASS_EXACT_FULL_NATIVE_MLP_LEARNED' and p['score_feature_pairs']==2067
  assert len(s['trials'])==3 and s['GPU_UUID']==original['GPU_UUID']
  assert s['stock_prediction_SHA']==read(REPORTS/'SPEED_mlp_learned.json')['stock_prediction_SHA']
  assert s['trained_checkpoint']['SHA256']==read(REPORTS/'ONLINE_mlp_learned.json')['trained_checkpoint']['SHA256']
  ratio=s['mean_full_scene_FPS']/original['mean_full_scene_FPS']
  cached=dict(report=ref(cache_path),exact_full_native_parity=ref(REPORTS/'LEARNED_MLP_FEATURE_CACHE_PARITY.json'),
   full_image_FPS=2*s['mean_full_scene_FPS'],FPS_ratio_to_GMT=ratio,efficiency_gate_pass=ratio>=.95,
   quality_reused_only_because_all_full_raw_predictions_identical=True)
  methods['mlp']['cached_deployment']=cached
  lines+=['',f"缓存MLP另报：完整图像FPS {cached['full_image_FPS']:.6f}，相对GMT {ratio:.6f}；95%门槛 {cached['efficiency_gate_pass']}。默认结果保留。完整2067次调用与raw输出严格一致，因此复用同一质量结果，不以改策略换加速。"]
 if methods:
  lines+=['','| 头 | 已正确接受式参考 | 保持率 | N01 | N10 | UNKNOWN目标 | UNKNOWN参考Bank/出生 |','|---|---:|---:|---:|---:|---:|---:|']
  for kind,r in methods.items():
   c=r['paired_current_state_counts'];rate=r['correct_accepted_reference_preservation_rate']
   rate_text=f'{100*rate:.6f}%' if rate is not None else 'UNDEFINED'
   lines.append(f"| {kind} | {c.get('reference_correct',0)} | {rate_text} | {c.get('N01',0)} | {c.get('N10',0)} | {c.get('UNKNOWN_target',0)} | {c.get('UNKNOWN_reference_Bank_or_birth',0)} |")
  lines+=['','该局部计数只对同一当前真实状态的原始GMT接受式MATCH作参考，原始拒绝后的Bank反事实未知。HOTA正号不代表已证明局部纠错机制，UNKNOWN覆盖及后续状态传播须分开解释。']
 lines+=['','主要协议是完整native raw；原min50/stock另报，不按结果换协议。TRAIN16是开发集，普通MLP/Attention结果一并保留。单种子差值、训练loss和ORACLE均不能证明显著性或独立结构优势。',
  '长期错误统计目前是回顾性描述量，accepted-reference N01/N10是在同一当前状态的局部比较，不能冒充整段干预因果传播。完整FPS来自GPU7三次真实图像推理，包括前端/关联/提交/最终stock；共享宿主机串行块存在时间负载差异。',
  '完整目标仍待联合质量/效率、普通控制器优势、必要种子和合法合格独立视频。SCIENTIFIC_SUCCESS=false。']
 (DOCS/'LEARNED_GATE_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
 (DOCS/'TRAINING_RESULTS.md').write_text('\n'.join(lines[:lines.index('| 完整冷启动 TRAIN16 | HOTA | ΔHOTA | AssA | CVIDF1 | IDSW | Frag | 完整图像FPS |')])+'\n\n完整闭环与效率结果见 LEARNED_GATE_RESULTS_ZH.md。\n')
 summary=dict(status=revision_status,training_metrics=ref(REPORTS/'TRAINING_METRICS.json'),methods=methods,
  all_three_quality_complete=len(methods)==3,all_three_speed_complete=len(methods)==3 and all(r['full_image_FPS'] is not None for r in methods.values()),
  scientific_success=False,independent_validation=False)
 save(REPORTS/'LEARNED_GATE_RESULTS.json',summary)
 decision=read(REPORTS/'GO_NO_GO.json');decision['single_learned_gate_revision'].update(status=revision_status,
  new_gradient_updates=updates,evidence=ref(REPORTS/'LEARNED_GATE_RESULTS.json'))
 decision['historical_rule_control_experiments_complete']=True
 decision['bounded_tracking_experiments_complete']=summary['all_three_quality_complete'] and summary['all_three_speed_complete']
 decision['learned_development_quality']={k:dict(delta_HOTA=r['delta']['HOTA'],joint_measured_quality_pass=r['joint_measured_quality_pass'],quality_gates=r['quality_gates']) for k,r in methods.items()}
 decision['learned_training']='SINGLE_BOUNDED_LEARNED_GATE_REVISION_'+revision_status
 positives=[k for k,r in methods.items() if r['joint_measured_quality_pass']]
 decision['current_measured_quality_positive_candidates']=positives
 if len(methods)==3 and 'mlp' in positives and 'cic' not in positives:
  decision['CIC_architecture_go_no_go']='NO_GO_CURRENT_FIXED300_DEVELOPMENT_SEED; retain ordinary MLP, do not force JEV or claim significance'
  if methods['mlp']['efficiency_gate_pass'] is False:
   decision['current_best_conservative_candidate']='Fixed300 MLP passes measured raw quality gates but default full-image FPS fails 95%; exact-cache deployment check is separate; seed and independent validation pending'
  elif methods['mlp']['efficiency_gate_pass'] is True:
   decision['current_best_conservative_candidate']='Fixed300 MLP passes measured raw quality and default full-image speed; seed, causal mechanism and independent validation pending'
  else:
   decision['current_best_conservative_candidate']='Fixed300 MLP passes measured raw quality gates; full-image FPS, seed reproduction and independent validation pending'
  if cached:
   decision['cached_MLP_efficiency_gate_pass']=cached['efficiency_gate_pass']
 early=REPORTS/'LEARNED_FIRST_DIVERGENCE_NATIVE.json'
 if early.exists():
  d=read(early);decision['first_native_discrepancy_audit']=ref(early)
  lines2=['# 首次输出分歧的原生状态配对核验','',
   '仅重跑TRAIN16前三个场景，两条轨迹均冷启动；不恢复旧JEV前缀、不启用旧JEV策略、不改变权重。前三场景预测分别与各自已保留完整轨迹严格一致。','',
   '| 当前关联前的共同输入 | 严格一致 |','|---|---|']
  for key,passed in d['common_input_checks'].items():lines2.append(f'| {key} | {passed} |')
  lines2+=['','| 运行 | 离线目标类别 | 原始接受ID | 修正接受ID | 实际提交ID | 新生身份 |','|---|---|---|---|---|---|']
  for kind,c in d['cases'].items():
   r=c['inspected_row'];lines2.append(f"| {kind} | {r['category']} | {r['base_accepted_ID']} | {r['modified_accepted_ID']} | {r['actual_native_ID']} | {r['actual_is_newborn']} |")
  lines2+=['','只有共同输入全部严格一致，才支持局部同状态比较。当前核验包含完整native状态/RNG、派生摘要、GTA原始分数、24维观测、候选及支持长度；记录了真实Bank/出生提交。',
   '该行GT为UNKNOWN，不能把ID52改成ID25计为已认证N01，也不能说已证明纠正了身份错误。没有执行该事件单独开启/关闭的完整视频干预，不能把HOTA增益归因于此事件或宣称已测真实因果传播时间。']
  (DOCS/'LEARNED_FIRST_DIVERGENCE_NATIVE.md').write_text('\n'.join(lines2)+'\n')
 save(REPORTS/'GO_NO_GO.json',decision)
 goal=read(REPORTS/'MASTER_GOAL.json');goal['gates']['ORDINARY_CONTROL_ADVANTAGE']='LEARNED_FAIR_COMPARISON_PENDING' if len(methods)<3 else 'SEE_LEARNED_GATE_RESULTS_NOT_SIGNIFICANCE_PROOF'
 goal['gates']['SPEED']='DEFAULT_LEARNED_TIMING_IN_PROGRESS' if not summary['all_three_speed_complete'] else 'COMPLETE_DEFAULT_LEARNED_TIMINGS_SEE_PER_METHOD_GATES'
 if cached:goal['gates']['CACHED_MLP_SPEED']='PASS_MEASURED' if cached['efficiency_gate_pass'] else 'FAIL_MEASURED'
 positives=[k for k,r in methods.items() if r['joint_measured_quality_pass']]
 if positives:
  goal['gates']['MATCHED_DEVELOPMENT_GAIN']='MEASURED_RAW_QUALITY_POSITIVE_'+','.join(positives)+'_SPEED_CAUSAL_SEEDS_INDEPENDENCE_NOT_PROVEN'
  goal['gates']['IDENTITY_STABILITY']='MEASURED_RAW_AND_DESCRIPTIVE_GATES_PASS_'+','.join(positives)+'_CAUSAL_PROPAGATION_NOT_ESTABLISHED'
 goal['gates']['LEARNED_GATE_SUPERVISION']='PASS_NATIVE_ZERO_AND_SUPERVISION_CONTRACT' if pre.exists() else 'PENDING_REAL_SUPERVISION_PRECHECK'
 save(REPORTS/'MASTER_GOAL.json',goal)
 speed=read(REPORTS/'SPEED_METRICS.json');speed['trained_neural_model_quality']='SEE_LEARNED_GATE_RESULTS' if methods else 'PENDING_LEARNED_FIXED_ENDPOINT_ONLINE'
 speed['learned_gate_methods']={k:{key:r[key] for key in ['full_image_FPS','FPS_ratio_to_GMT','efficiency_gate_pass','speed_report']} for k,r in methods.items()}
 if cached:speed['learned_cached_MLP']=cached
 save(REPORTS/'SPEED_METRICS.json',speed)
 with (DOCS/'EFFICIENCY.md').open('a') as f:
  f.write('\n训练后 learned gate 的完整 FPS 另见 LEARNED_GATE_RESULTS_ZH.md；旧零残差原型和旧 conflict 控制器结果保持原语义，不能将其计时当作新 learned gate 的计时。\n')
  profile=REPORTS/'LEARNED_MLP_COMPONENT_PROFILE.json'
  if profile.exists():
   d=read(profile);f.write('\n训练后 MLP 的 GPU8、24场景帧同步边界诊断（非完整FPS，不外推整段）：\n\n| 部分 | p50 ms | p95 ms |\n|---|---:|---:|\n')
   for name,r in d['component_ms'].items():f.write(f"| {name} | {r['p50_p95_max'][0]:.4f} | {r['p50_p95_max'][1]:.4f} |\n")
   f.write('\n该前缀中特征构建耗时高于小头计算；同步边界改变执行行为，只用于定位后续效率检查方向。实际效率结论仍只由GPU7完整图像计时给出。\n')
 if updates:
  reviewer=DOCS/'REVIEWER_ANALYSIS.md';review_text=reviewer.read_text()
  review_text=review_text.replace('当前 MLP/Attention/CIC 仅有零残差原型、参数与工程等价/开销证据；没有训练、没有性能显著性，也不能声称优于它们。','早期 conflict 模式 MLP/Attention/CIC 只有零残差原型证据；新 learned gate 的固定300步同预算训练及闭环评测另见 LEARNED_GATE_RESULTS_ZH.md。训练本身不证明性能显著性或独立架构优势。')
  reviewer.write_text(review_text)
 report=DOCS/'FINAL_RESEARCH_REPORT_ZH.md';text=report.read_text()
 text=text.replace('当前仍为0真实训练更新','该修订真实训练更新总数以 TRAINING_METRICS.json 为准')
 if updates:
  text=text.replace('但没有执行训练，新增梯度更新为0。','早期 conflict 原型未训练；新 learned gate 的实际训练及结构化 hinge loss 见下文补查。')
  text=text.replace('MLP/Attention/CIC 没有通过冻结 TRAIN 信号前置条件，不进行盲目训练或架构消融。','旧 conflict 模式未通过冻结 Rule 信号条件；一次经原生零残差与认证监督预检查的 learned gate 修订另列，未扩大架构搜索。')
  text=text.replace('目前所有正向结果来自未训练的普通控制器','此前 Rule/Threshold/Confidence 的正向结果来自未训练的普通控制器')
 text+='\n## 学习式门控修订的当前证据\n\n阶段 '+revision_status+f'；真实 optimizer 更新总数 {updates}。三个头各自的固定终点、监督认证和完整闭环结果见 [LEARNED_GATE_RESULTS_ZH.md](LEARNED_GATE_RESULTS_ZH.md)。旧 Rule 失败结论不变；未证明独立科学成功。\n'
 report.write_text(text)
 from gmt_jev_current_report import main as current_report
 current_report()
 print('LEARNED_EVIDENCE_REFRESHED',revision_status,updates,len(methods),flush=True)

if __name__=='__main__':main()
