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
   '时间留出桶不参与训练；GT只作离线认证，UNKNOWN、不唯一、阈值不可达及无联合有界纠错见证的错误行被mask。原生完整矩阵仍参与训练求解，UNKNOWN候选竞争不能忽略。']
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
 decision['learned_training']='SINGLE_BOUNDED_LEARNED_GATE_REVISION_'+revision_status
 save(REPORTS/'GO_NO_GO.json',decision)
 goal=read(REPORTS/'MASTER_GOAL.json');goal['gates']['ORDINARY_CONTROL_ADVANTAGE']='LEARNED_FAIR_COMPARISON_PENDING' if len(methods)<3 else 'SEE_LEARNED_GATE_RESULTS_NOT_SIGNIFICANCE_PROOF'
 goal['gates']['LEARNED_GATE_SUPERVISION']='PASS_NATIVE_ZERO_AND_SUPERVISION_CONTRACT' if pre.exists() else 'PENDING_REAL_SUPERVISION_PRECHECK'
 save(REPORTS/'MASTER_GOAL.json',goal)
 speed=read(REPORTS/'SPEED_METRICS.json');speed['trained_neural_model_quality']='SEE_LEARNED_GATE_RESULTS' if methods else 'PENDING_LEARNED_FIXED_ENDPOINT_ONLINE'
 speed['learned_gate_methods']={k:{key:r[key] for key in ['full_image_FPS','FPS_ratio_to_GMT','efficiency_gate_pass','speed_report']} for k,r in methods.items()}
 save(REPORTS/'SPEED_METRICS.json',speed)
 with (DOCS/'EFFICIENCY.md').open('a') as f:
  f.write('\n训练后 learned gate 的完整 FPS 另见 LEARNED_GATE_RESULTS_ZH.md；旧零残差原型和旧 conflict 控制器结果保持原语义，不能将其计时当作新 learned gate 的计时。\n')
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
 print('LEARNED_EVIDENCE_REFRESHED',revision_status,updates,len(methods),flush=True)

if __name__=='__main__':main()
