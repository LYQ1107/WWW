"""Matched temporal diagnostics of actual native stock outputs, no inference/training."""
import collections
from gmt_jev_common import *
from gmt_jev_evaluate import duration
from gmt_jev_past_anchor_audit import load,audit,synthetic_checks

def main():
 from jev_phase7_offline import IdentityEvaluator
 e=IdentityEvaluator(16);checks=synthetic_checks()
 duplicate={i:{g for g,n in collections.Counter(int(a['instance_id']) for a in gt).items() if n>1} for i,gt in e.gt.items()}
 image_by_key={(int(i['frame_id'])-1,int(i['view_id'])-1):k for k,i in e.images.items()}
 sources={'original':read(REPORTS/'ORIGINAL_GMT_STOCK_METRICS.json')['predictions']}
 sources.update({k:read(REPORTS/f'ONLINE_{k}.json')['stock_predictions'] for k in ['threshold','rule','confidence']})
 cases={}
 for kind,source in sources.items():
  pred=load(source);aligned=e.align(pred);labels={key:r['gt'] for key,r in aligned.items()};masked=0
  for key,g in labels.items():
   if g in duplicate[image_by_key[key[:2]]]:labels[key]=None;masked+=1
  cases[kind]=dict(predictions=source,retrospective_global_map=duration(pred),
   past_certified_anchor=audit(pred,labels,e.images),duplicate_GT_observations_masked=masked)
 for r in cases.values():
  r['retrospective_duration_ratio_to_original']=r['retrospective_global_map']['identity_inconsistency_duration_camera_frames']/cases['original']['retrospective_global_map']['identity_inconsistency_duration_camera_frames']
  denominator=cases['original']['past_certified_anchor']['wrong_person_observed_duration_camera_frames']
  r['past_anchor_duration_ratio_to_original']=r['past_certified_anchor']['wrong_person_observed_duration_camera_frames']/denominator if denominator else None
 result=dict(status='COMPLETE_MATCHED_NATIVE_STOCK_TEMPORAL_DIAGNOSTIC',methods=cases,video=16,
  no_new_inference=True,no_gradient_updates=True,GT_actor_inputs=False,post_hoc_diagnostic=True,
  input_convention='Actual unchanged native min_track_len50 and postprocessing, not synthetic removal or raw metrics.',
  limitations=['Final stock outputs depend on full-video track-length filtering, hence are not causal online evidence',
   'Past anchors use earlier scene frames of the retained final-output series; unknown/gaps are censored, not repaired',
   'Retrospective global identity mapping and historical anchor identity persistence are distinct diagnostics',
   'Post-hoc diagnostic cannot change the preregistered primary raw outcome or select a favorable evaluation protocol'],
  synthetic_semantic_checks=checks,scientific_success=False,source=binding())
 save(REPORTS/'STOCK_IDENTITY_TEMPORAL_AUDIT.json',result)
 lines=['# 原生最终输出的长期身份诊断','','固定原生 min_track_len=50 与后处理，各方法使用各自完整视频真实输出；没有新推理或训练。','',
  '| 方法 | 回顾性身份不一致相机帧 | 相对 GMT | 历史锚错误人物相机帧 | 相对 GMT | 未认证 owner 观测 |', '|---|---:|---:|---:|---:|---:|']
 for kind,r in cases.items():
  anchor=r['past_certified_anchor'];ratio=r['past_anchor_duration_ratio_to_original'];ratio_text=f'{ratio:.4f}' if ratio is not None else 'UNDEFINED'
  lines.append(f"| {kind} | {r['retrospective_global_map']['identity_inconsistency_duration_camera_frames']} | {r['retrospective_duration_ratio_to_original']:.4f} | {anchor['wrong_person_observed_duration_camera_frames']} | {ratio_text} | {anchor['counts'].get('UNKNOWN_UNCERTIFIED_OWNER_observations',0)} |")
 lines+=['','回顾性映射沿用原先 duration 定义；历史锚采用先前场景至少3次已知观测且覆盖率>=80%的首个人物锚，重复 GT 人物标注屏蔽。遮挡/未知/未认证 owner 为删失，不能当作修复。',
  '', '最终输出本身依赖全视频轨迹长度过滤，所以即使锚只用先前输出帧，该表也不能证明在线因果错误传播或未来安全。结果后诊断不替代主要 raw 协议，也不将删失区间或未知覆盖变化解释成纠错。详细覆盖、最长区间和输入SHA见 JSON。']
 (DOCS/'STOCK_IDENTITY_TEMPORAL_AUDIT_ZH.md').write_text('\n'.join(lines)+'\n')
 print('GMT_STOCK_IDENTITY_AUDIT_COMPLETE',{k:(r['retrospective_duration_ratio_to_original'],r['past_anchor_duration_ratio_to_original']) for k,r in cases.items()},flush=True)

if __name__=='__main__':main()
