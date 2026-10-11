"""Publish fixed-profile parity attribution without promoting teacher identity claims."""
import os
os.environ.setdefault('JEV_PHASE21_OUT','/home/liuyeqiang/WWW_jev_phase21_runtime/20261011_v2')
from jev_phase21_common import *
import collections,statistics

def main():
 protect();r=read(OUT/'RESULT.json');pre=read(REPORTS/'PREREGISTRATION.json')
 assert len(r['tests'])==72 and len(r['latencies'])==80
 assert r['binding']['preregistration']==ref(REPORTS/'PREREGISTRATION.json')
 actor=r['binding']['source_commit']
 first_failure=read(REPORTS/'ATTEMPT1_FAILURE.json')
 for attempt_binding in [first_failure['binding']]:
  for n,h in attempt_binding['source_SHA256'].items():
   assert hashlib.sha256(subprocess.check_output(['git','show',attempt_binding['source_commit']+':'+n],cwd=ROOT)).hexdigest()==h
 for n,h in r['binding']['source_SHA256'].items():
  raw=subprocess.check_output(['git','show',actor+':'+n],cwd=ROOT)
  assert hashlib.sha256(raw).hexdigest()==h,(actor,n)
  assert sha(ROOT/n)==h,n
 for n,h in read(PRIOR/'reports/JEV_PHASE20/ARTIFACT_SHA_MANIFEST.json')['files'].items():assert sha(PRIOR/n)==h,n
 assert sha(pre['previous']['path'])==pre['previous']['SHA256']
 prior_gate=Path('/home/liuyeqiang/WWW_jev_phase19/reports/JEV_PHASE19/LEARNING_ELIGIBILITY.json')
 gate=read(prior_gate);assert not gate['qualified'] and (gate['safe_TRAIN'],gate['safe_RESERVED'])==(3,1)
 # Verify all original cases, never select the profiles' most favourable groups.
 old=read(pre['previous']['path']);by_old={(t['variant'],t['group']):t for t in old['tests']}
 indexes={p['name']:{(t['variant'],t['group']):t for t in r['tests'] if t['profile']==p['name']} for p in pre['profiles']}
 old_exact=[]
 for key,t in indexes['FP16_DEFAULT'].items():
  orig=by_old[key];assert t['image']==orig['source_image']
  paths=[p for p in t['paths'] if p['path']!='vision_cache']
  old_exact.append(paths==orig['paths'] and t['option_reverse_decision_exact']==orig['option_reverse_decision_exact'])
 baseline=indexes['FP16_DEFAULT'];cross=[]
 for name,idx in indexes.items():
  changed_questions=0;changed_who=0;max_delta=0.
  for k,t in idx.items():
   a=baseline[k]['full_forward_probabilities'];b=t['full_forward_probabilities']
   for i,(pa,pb) in enumerate(zip(a,b)):
    same=max(range(len(pa)),key=pa.__getitem__)==max(range(len(pb)),key=pb.__getitem__)
    changed_questions+=not same;changed_who+=(i==0 and not same)
    max_delta=max(max_delta,max(abs(x-y) for x,y in zip(pa,pb)))
  cross.append(dict(profile=name,reference='FP16_DEFAULT',changed_full_forward_question_decisions=changed_questions,
   question_total=120,changed_full_forward_WHO_decisions=changed_who,WHO_total=24,max_probability_delta=max_delta))
 preferred=next((p['name'] for p in pre['profiles'] if r['summary'][p['name']]['all_paths_pass']),None)
 all_decisions={n:sum(p['decisions_exact'] for t in indexes[n].values() for p in t['paths']) for n in indexes}
 save(REPORTS/'VISUAL_PROFILE_ATTRIBUTION.json',dict(status='FIXED_PROFILE_NUMERICAL_ATTRIBUTION_COMPLETE',
  actor_commit=actor,original_default_reproduced_exact_groups=sum(old_exact),original_total=24,
  per_profile_summary=r['summary'],decisions_exact_path_checks=all_decisions,path_checks_per_profile=96,
  full_forward_cross_profile=cross,compute_only_preferred_profile=preferred,
  failure_reasons={n:dict(logit_exceeds_tolerance=sum(a['max_abs_valid_logit_delta']>.05 for t in idx.values() for a in t['paths']),
   probability_exceeds_tolerance=sum(a['max_abs_probability_delta']>.02 for t in idx.values() for a in t['paths']),
   decisions_differ=sum(not a['decisions_exact'] for t in idx.values() for a in t['paths'])) for n,idx in indexes.items()},
  teacher_identity_superiority='NOT_ESTABLISHED',native_serving_cache='NOT_IMPLEMENTED_OR_QUALIFIED',
  identification_limits=['FP16_DEFAULT versus FP16_MATH changes backend only.',
   'FP16_MATH versus FP32_MATH changes runtime precision only, including any adapter dtype recorded per case.',
   'Default backend flags admit available SDPA kernels; actual per-operation kernel traces were not collected.',
   'Finite 12-group audit does not establish universal floating-point equivalence.',
   'Option reversal changes the prompt; a passing same-prompt cache audit cannot establish presentation invariance.']))
 latency=dict(status='FIXED_MICROBENCHMARK_COMPLETE',measurements=r['latencies'],
  environment=r['environment'],timing_samples=2,p95=None,source='Existing decoded montage and repeated five question templates',
  not_native_MOT=True,not_dynamic32identity_candidates=True,not_image_IO_or_live_serving=True,
  speedups=[])
 for profile in pre['latency']['profiles']:
  for variant in pre['variants']:
   for n in pre['latency']['N']:
    rows={x['path']:x for x in r['latencies'] if x['profile']==profile and x['variant']==variant and x['N']==n}
    a=rows['independent_serial']['p50_total_ms'];b=rows['independent_batch']['p50_total_ms'];c=rows['shared_prefix_batch']['p50_total_ms']
    latency['speedups'].append(dict(profile=profile,variant=variant,N=n,serial_ms=a,batch_ms=b,shared_ms=c,
      serial_over_shared=a/c,batch_over_shared=b/c,shared_peak_MiB=rows['shared_prefix_batch']['peak_allocated_MiB'],
      batch_peak_MiB=rows['independent_batch']['peak_allocated_MiB']))
 save(REPORTS/'VISUAL_EXECUTION_LATENCY.json',latency)
 released={p.name:read(p) for p in OUT.glob('RELEASE_*.json')}
 assert len(released)==3 and all(x['free_MiB']>=26000 for x in released.values()),released
 attempts=[Path('/home/liuyeqiang/WWW_jev_phase21_runtime/20261011_v1'),OUT]
 size=sum(p.stat().st_size for d in attempts for p in d.rglob('*') if p.is_file() and not p.is_symlink())
 assert size<=64*2**20
 audit=dict(status='ENGINEERING_AUDIT_COMPLETE',actor_commit=actor,source_files_verified_against_actor_commit=True,
  frozen_Phase20_publication_files_verified=27,original_Phase17_cache_NO_GO_preserved=True,
  weight_files_SHA_verified=len(r['verified_existing_weights']),verified_weights=r['verified_existing_weights'],
  protocol_unchanged_SHA=sha(REPORTS/'PREREGISTRATION.json'),GPU_workers=1,
  runtime_total_MiB=size/2**20,home_free_GiB=shutil.disk_usage(OUT).free/2**30,model_release_measurements=released,
  original_failure_retained=ref(REPORTS/'ATTEMPT1_FAILURE.json'),engineering_repairs=1,
  successful_retry_seconds=r['seconds'],new_training=False,new_native_actions=0,new_models=0,
  sealed20_21_22_accessed=False,official_TEST=False,Full24=False,
  all_72_groups_token_position_KV_no_alias_checks=True,learning_gate_preserved=ref(prior_gate))
 save(REPORTS/'GPU_EXECUTION.json',read(OUT/'GPU_EXECUTION.json'))
 audit['GPU_execution']=ref(REPORTS/'GPU_EXECUTION.json')
 save(REPORTS/'ENGINEERING_AND_DATA_AUDIT.json',audit)
 decision=dict(status='BOUNDED_EXISTING_VISUAL_EXECUTION_AUDIT_COMPLETE',master_goal_complete=False,
  compute_only_preferred_profile=preferred,numerical_cache_parity='PASS_FIXED_POPULATION' if preferred else 'NO_GO_ALL_FIXED_PROFILES',
  option_order_robustness='PASS_FIXED_POPULATION' if preferred and r['summary'][preferred]['order_exact']==24 else 'NO_GO_OPTION_ORDER',
  teacher_identity_value='NOT_ESTABLISHED',trained_candidate_controller='NOT_RUN',native_end_to_end_deployment='NOT_QUALIFIED',
  independent_safe_supervision=dict(TRAIN=3,RESERVED=1,required_TRAIN=8,required_RESERVED=3),
  new_training=False,scientific_success=False,paper_readiness='NOT_READY',
  next_required='Independent manually valid supervision source qualification; Phase20 source proposal remains pending user scope/access reply.',
  prior_gate_unchanged=ref(prior_gate),official_TEST=False,Full24=False)
 save(REPORTS/'FINAL_GO_NO_GO.json',decision)
 table=[]
 for n,s in r['summary'].items():
  table.append('| '+' | '.join([n,str(s['path_passes']['vision_cache']),str(s['path_passes']['independent_batch']),str(s['path_passes']['prefix_share']),str(s['path_passes']['prefix_share_batch']),f"{s['max_logit_delta']:.8g}",f"{s['order_exact']}/24"] )+' |')
 n32=[x for x in latency['speedups'] if x['N']==32]
 lt=['| '+' | '.join([x['profile'],x['variant'],f"{x['serial_ms']:.1f}",f"{x['batch_ms']:.1f}",f"{x['shared_ms']:.1f}",f"{x['serial_over_shared']:.2f}",f"{x['batch_over_shared']:.2f}"])+' |' for x in n32]
 text=f'''# Phase XXI — 既有 Visual JEV 执行一致性归因

本轮仅复用Phase XVII的12个图像/问题组与两种既有权重，完成三个事前冻结执行配置的72个group-variant、288个路径比较。模型和样本不变，没有新数据、训练或原生动作。计算路径推荐为 **{preferred or '无合格配置'}**；这只是有限样本的数值一致性资格，身份价值、选项顺序鲁棒性与native部署须分别评价。

## 原问题与固定对照

Phase XVII的FP16审计保留原始NO_GO：72个路径比较中22个失败，但决策全部一致，主要失败条件是有效选项raw logit绝对差超过0.05。本轮保持logit≤0.05、概率差≤0.02、决策完全一致的原门槛，比较FP16默认SDPA、FP16数学SDPA、FP32数学SDPA。默认→数学只改变后端；数学FP16→FP32只改变执行精度。没有逐算子kernel trace，因此不把默认配置说成所有算子均使用某个特定kernel。

每组均验证真实完整processor token、公共前缀位置一致及KV fork无别名；图像来自同一原始pair/哈希。五个问题为WHO、OWNER_EVIDENCE、AMBIGUITY、CROSS_VIEW、FALSE_PREMISE。两种权重是既有Qwen3-VL4B与公布answer LoRA，保留原LM head，不训练新任务头。原始24组中 **{sum(old_exact)}/24** 的三个旧路径和顺序判定精确复现。

| 配置 | vision-cache PASS /24 | batch PASS /24 | prefix-share PASS /24 | prefix-share-batch PASS /24 | 最大logit差 | WHO反序语义决策一致 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

在这批固定样本上，两个FP16配置均未全部通过，FP32数学配置全部通过。这支持将此前的数值门槛失败归因到低精度执行，而非另一个token、位置或KV语义修复；有限样本不证明全局等价，也不说明所有数值差异仅来自某一种kernel。完整前向的跨配置决策变化及概率差记录在VISUAL_PROFILE_ATTRIBUTION.json。相比默认FP16，FP16数学与FP32数学的120个完整前向决策均无变化，但最大概率差分别约0.0618和0.0562，因而不能将旧配置的confidence校准自动视为新配置已验证。这些跨配置量不能由缓存对照中决策一致直接推出。所有配置完整披露，不按输出换样本、调容差或保留有利结果。

选项反序会改变实际提示词；即使token/cache完全正确，模型仍可能改变同一身份选项的判断。这是候选表达的敏感性，不是相同提示词的KV计算错误。数值一致性通过也不升级此前教师身份识别价值或native候选控制器资格。有限12组不是独立新验证集。

## 实际成本

每个配置/权重/N/路径先warmup一次，再计两次GPU同步执行。报告总batch延迟、两次原值/中位数/最大值、峰值显存；prefill与KV fork单独计时。两次测量不能估计p95，p95明确为null。N32为同一已解码蒙太奇的重复问题压力测试，不是32个真实动态身份候选，也不是在线MOT吞吐或包含图像IO的服务延迟。

| 配置 | 权重 | N32串行总ms | batch总ms | 共享总ms | 串行/共享 | batch/共享 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lt)}

## 工程失败与复现

第一次在完成FP16_DEFAULT的24组与全部计时后，下一配置准入失败：模型仍由最后的bound method及parameter引用保留，显存未释放。原FAILURE、部分结果和原提交5084c075完整保留。唯一普通工程修复在b1b78b40：显式释放model、method、parameter，保留所有科学设置，在独立v2目录重跑。三个配置释放后均实测free memory≥26,000MiB；少量剩余allocated memory按原值记录，不声称为零。旧结果不覆盖。不是架构科研失败，也没有放宽26,000MiB准入要求。

成功actor固定于{actor}，源码SHA与该Git提交逐项核验；协议SHA {sha(REPORTS/'PREREGISTRATION.json')}。三份权重文件SHA实测一致；Phase XX全部27个发布文件及Phase XVII原审计不变。单GPU2，成功重跑约{r['seconds']:.1f}秒，两轮留存约{size/2**20:.3f}MiB，home空闲{audit['home_free_GiB']:.2f}GiB。失败日志、精简JSON与代码推GitHub，图像/模型留服务器。

## 主科研目标

本轮解决或限定的是执行一致性，未建立新的身份识别、全视频关联收益或架构独立优势。现有安全监督仍为3TRAIN＋1预留，低于8＋3门槛；训练NOT_RUN。原生cache服务没有在此轮实现或验证，冷启动、跨状态失效、活跃图库与真实图像流水线均不能借这次microbenchmark宣称完成。论文仍NOT_READY，MASTER继续ACTIVE/PIVOT_REQUIRED。

下一步沿已推送Phase XX独立来源方案进行数据资格审计，等待新增来源范围/合法数据路径回复；本轮没有读取新增数据、sealed20/21/22或官方TEST。复核入口：PREREGISTRATION、VISUAL_EXECUTION_PROFILE_PARITY、VISUAL_PROFILE_ATTRIBUTION、VISUAL_EXECUTION_LATENCY、ENGINEERING_AND_DATA_AUDIT、FINAL_GO_NO_GO及ARTIFACT_SHA_MANIFEST。
'''
 (ROOT/'docs/PHASE21_FINAL_RESEARCH_REPORT.md').write_text(text)
 goal=read(MASTER/'MASTER_GOAL.json');goal.update(current_phase_status='VISUAL_EXECUTION_AUDIT_COMPLETE_SOURCE_QUALIFICATION_REQUIRED',
  master_status='PIVOT_REQUIRED',active_hypotheses=[],next_experiment='NEW_DATA_SOURCE_QUALIFICATION',scientific_SUCCESS=False,paper_readiness='NOT_READY')
 goal['completed_supported_hypotheses']=[x for x in goal.get('completed_supported_hypotheses',[]) if x['id']!='E21_H1']
 goal['completed_supported_hypotheses'].append(dict(id='E21_H1',result='Fixed numerical profile attribution complete; compute-only preferred '+str(preferred)+'; no teacher/native gain claim'))
 save(MASTER/'MASTER_GOAL.json',goal)
 reg=read(MASTER/'EXPERIMENT_REGISTRY.json');reg['experiments'][0].update(status='COMPLETE',returncode=0,actual_actor_commit=actor,
  progress_path=str(OUT/'PROGRESS.json'),result_path=str(OUT/'RESULT.json'),evidence=[ref(REPORTS/'FINAL_GO_NO_GO.json'),ref(REPORTS/'VISUAL_EXECUTION_PROFILE_PARITY.json')])
 save(MASTER/'EXPERIMENT_REGISTRY.json',reg)
 save(MASTER/'NEXT_ACTION.json',dict(status='PIVOT_REQUIRED_DATA_ACCESS_OR_SCOPE',scientific_success=False,
  proposal=str(ROOT/'docs/PHASE20_NEXT_DATA_SCOPE_PROPOSAL.md'),next='Qualify independent manually-valid data sources; pending existing user question, no implicit image/training authorization.',
  no_training_or_image_downloads=True,blocker='Independent safe supervision3TRAIN/1RESERVED; no established independent source access.'))
 ev=read(MASTER/'EVIDENCE_REGISTRY.json');ev['entries']=[x for x in ev['entries'] if '/reports/JEV_PHASE21/' not in x['path']];ev['entries'].extend([ref(REPORTS/'FINAL_GO_NO_GO.json'),ref(REPORTS/'VISUAL_PROFILE_ATTRIBUTION.json')]);save(MASTER/'EVIDENCE_REGISTRY.json',ev)
 note='\n\nPhase XXI: fixed original12group/2weight execution audit, unchanged .05/.02/exact decision gates; compute-only preferred '+str(preferred)+'. Option-order, independent teacher identity value and native deployment remain separate. One bound-reference memory repair; original failed attempt preserved. No new training/native actions/data.\n'
 p=MASTER/'RESEARCH_DECISIONS.md'
 if note not in p.read_text():
  with p.open('a') as f:f.write(note)
 for n in ['WWW_JEV_RESEARCH_ROADMAP.md','WWW_JEV_PAPER_READINESS.md','WWW_2027_SUBMISSION_READINESS.md']:
  p=ROOT/'docs'/n
  if note not in p.read_text():
   with p.open('a') as f:f.write(note)
 protect();print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
