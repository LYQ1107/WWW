"""Assemble native measured results; no inference, GT editing, or tuning."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,csv,hashlib,shutil,sys,subprocess
import numpy as np
repo=Path('/data1/liuyeqiang/WWW_worldmot_v4_c1_fulltest_20261026');root=Path('/data1/liuyeqiang/worldmot_v4_c1_fulltest_outputs_20261026');rt=Path('/data1/liuyeqiang/worldmot_v4_c1_fulltest_runtime_20261026')
sys.path[:0]=[str(repo/'worldmot'),str(repo/'worldmot/analysis')]
from worldmot.runtime.world_object_v4_test_contract import require_scope,completed_scene,sha256,write_json,LEGAL_TEST,FULL_SHA
from worldmot.runtime.experiment_authorization import source_fingerprint
p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
out=repo/'worldmot/results';docs=repo/'worldmot/docs';fp=source_fingerprint();scope,_=require_scope(root/'scope_v4_c1_TEST.json',root/'authorization_v4_c1_TEST.json','diagnostics')
assert fp=='9d5ad96c9e5ba1dea5edfdaf6e98a4563fc6dea3ce33f41332f8f939efd2be3c'
coverage=json.loads((root/'v4_c1_fulltest_coverage.json').read_text());assert coverage['status']=='COMPLETE' and coverage['timestamps']==29019 and coverage['candidate_count']==548583
for m in coverage['scenes']:completed_scene(root/f"predictions/V4_C1/scene_{m['scene']:02d}",scope,m['scene'])
metrics=json.loads((root/'evaluation/V4_C1/metrics.json').read_text());attribution=json.loads((root/'v4_c1_identity_attribution.json').read_text());state=json.loads((rt/'controller_state.json').read_text())
assert metrics['status']=='COMPLETE_VALID_SCOPE' and attribution['status']=='COMPLETE'
audit=json.loads((Path(scope['r2_root']).parent/'fulltrain_20261022/audit/data_audit.json').read_text())['test'];scene_names={s['scene']:s['name'] for s in audit['scenes']}
assert set(metrics['evaluated_scenes'])=={scene_names[s] for s in LEGAL_TEST} and not metrics['raw_GT_modified'] and not metrics['custom_metrics_used']
assert len(metrics['coverage'])==22 and sum(x['frames'] for x in metrics['coverage'])==29019
for item in metrics['coverage']:assert item['predictions_sha256']==next(x['tracks_sha256'] for x in coverage['scenes'] if x['scene']==item['scene'])
assert len(state['archives'])==22 and all('VERIFIED' in v['status'] for v in state['archives'].values())
def read(p):return json.loads(p.read_text())
def flatten(d):
 m=dict(d['TrackEval']['combined']);m.update({k:d['VisionTrack_MATLAB']['combined'][k] for k in ['CVIDF1','CVMA']});m['MATLAB_CLEAR_IDSW']=d['VisionTrack_MATLAB']['combined']['CLEAR_counts']['id_switches'];return m
baselines={
 'GMT Native':read(root/'input_provenance/GMT_NATIVE_OFFICIAL_official_metrics.json'),
 'GMTDet + V2':read(root/'input_provenance/GMTDET_V2_PRIMARY_official_metrics.json'),
 'Full No-Geometry V3':read(root/'input_provenance/FULL_NO_GEOMETRY_V3_official_metrics.json'),
 'Full V3 + Omega':read(root/'input_provenance/FULL_WORLD_OBJECT_V3_official_metrics.json')}
tool_comparability={}
new_driver=(repo/'worldmot/scripts/official_worldmot_metrics.m').read_text()
old_driver=subprocess.check_output(['git','show','7148803:worldmot/scripts/official_worldmot_metrics.m'],cwd=repo)
without_progress=''.join(line for line in new_driver.splitlines(True) if not any(tag in line for tag in ['WORLDMOT_OFFICIAL_JOB','partial=struct','pfid=fopen']))
assert without_progress.encode()==old_driver
for name,b in baselines.items():
 assert b['status']=='COMPLETE_VALID_SCOPE' and set(b['evaluated_scenes'])==set(metrics['evaluated_scenes'])
 assert b['TrackEval']['tool_sha256']==metrics['TrackEval']['tool_sha256']
 native=metrics['VisionTrack_MATLAB']['toolkit_sha256'];previous=b['VisionTrack_MATLAB']['toolkit_sha256']
 assert native and all(previous.get(k)==v for k,v in native.items())
 extra=sorted(set(previous)-set(native));assert all(k.startswith('utils/external/dollar/toolbox/detector/') for k in extra)
 old_hash=b['VisionTrack_MATLAB']['adapter_sha256'];new_hash=metrics['VisionTrack_MATLAB']['adapter_sha256']
 assert old_hash in {new_hash,hashlib.sha256(old_driver).hexdigest()}
 tool_comparability[name]={'TrackEval_29_source_hashes_equal':True,'MATLAB_47_native_tracking_dependency_hashes_equal':True,'additional_old_hash_inventory_detector_files':extra,
 'MATLAB_driver_byte_identical':old_hash==new_hash,'MATLAB_driver_difference_only_progress_logging':old_hash!=new_hash,'MATLAB_old_driver_SHA':old_hash,'MATLAB_current_driver_SHA':new_hash,
 'historical_MOT_text_format':'.2f' if extra else '.17g','current_MOT_text_format':'.17g','interpretation':'Full V3/No-Geometry direct controls share current wrapper and exact precision; GMT/V2 are published system baselines. Old driver differs only in stdout and partial-result progress writes; native tracking dependencies unchanged.'}
current=flatten(metrics);values={name:flatten(b) for name,b in baselines.items()};values['V4 C1']=current
columns=['HOTA','DetA','AssA','IDF1','CVIDF1','CVMA','MOTA','IDSW','Frag','CLR_TP','CLR_FP','CLR_FN','MATLAB_CLEAR_IDSW']
deltas={name:{k:current[k]-v[k] for k in columns} for name,v in values.items() if name!='V4 C1'}
with (out/'v4_c1_final_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,['method',*columns,'scope'],lineterminator='\n');w.writeheader()
 for name,v in values.items():w.writerow({'method':name,**{k:v[k] for k in columns},'scope':'original GT legal14 full TEST; native TrackEval and VisionTrack MATLAB'})
with (out/'v4_c1_differences.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,['reference',*columns],lineterminator='\n');w.writeheader()
 for name,v in deltas.items():w.writerow({'reference':name,**v})
shutil.copyfile(root/'v4_c1_fulltest_coverage.json',out/'v4_c1_fulltest_coverage.json')
shutil.copyfile(root/'v4_c1_identity_attribution.json',out/'v4_c1_identity_attribution.json')
base_proofs={name:{'local_snapshot_sha256':sha256(root/'input_provenance'/filename),'original':str(path)} for name,filename,path in [
 ('GMT Native','GMT_NATIVE_OFFICIAL_official_metrics.json','/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022/test_evaluation/gmt_native/metrics.json'),
 ('GMTDet + V2','GMTDET_V2_PRIMARY_official_metrics.json','/data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/test_gmt_v2_evaluation/joint_seed20261022_legal14_complete/metrics.json'),
 ('Full No-Geometry V3','FULL_NO_GEOMETRY_V3_official_metrics.json','/data1/liuyeqiang/nas/A100/WorldMOT/worldobject_v3_birthfix_r2_20261025/evaluation/FULL_NO_GEOMETRY_V3/metrics.json'),
 ('Full V3 + Omega','FULL_WORLD_OBJECT_V3_official_metrics.json','/data1/liuyeqiang/nas/A100/WorldMOT/worldobject_v3_birthfix_r2_20261025/evaluation/FULL_WORLD_OBJECT_V3/metrics.json')]}
bundle={'status':'COMPLETE_OFFICIAL_LEGAL14_TEST','variant':'C1','source_sha256':fp,'implementation_commit':'2020c24362bbc1bef3197e48b7a7d61a4ade77db','checkpoint_sha256':FULL_SHA,
 'inference_scene_count':22,'inference_timestamps':29019,'RGB_inputs':58038,'candidates':548583,'candidate_differences':0,'ROI_mutations':0,
 'official_scene_ids':list(LEGAL_TEST),'blocked8_scene_ids':[s for s in range(1,23) if s not in LEGAL_TEST],
 'original_GT_modified':False,'original_GT_annotations_sha256':attribution['original_GT_sha256'],'new_training':False,'TEST_parameter_selection':False,'exploratory_TEST_previously_informed_design':True,
 'TrackEval':metrics['TrackEval'],'VisionTrack_MATLAB':metrics['VisionTrack_MATLAB'],
 'official_scope':'14 original-GT-legal complete TEST videos; no cleaned invalid GT in official scores',
 'source_metrics_sha256':sha256(root/'evaluation/V4_C1/metrics.json'),'baselines':values,'baseline_proofs':base_proofs,'current_minus_reference':deltas,
 'shared_native_tracking_dependency_hashes_equal_to_baselines':True,'baseline_tool_comparability':tool_comparability,'new_full_training_authorized':False,'next_TEST_search_authorized':False}
write_json(out/'v4_c1_official_test.json',bundle)
# Cached execution time includes checks/output; archive time is separate.
all_rows=[];runtime_scenes={}
for m in coverage['scenes']:
 p=root/f"predictions/V4_C1/scene_{m['scene']:02d}/timings.jsonl";rows=[json.loads(line) for line in p.open()];assert len(rows)==m['timestamps'];all_rows+=rows
 runtime_scenes[str(m['scene'])]={k:m[k] for k in ['timestamps','candidate_count','execution_device','head_precision','started_UTC','completed_UTC','worker_wall_s','scene_loop_wall_s','cached_paired_times_per_s','peak_RSS_bytes','GPU_peak_allocated_bytes','timing_sums','motion_call_counts']}
keys=sorted(set().union(*(r.keys() for r in all_rows))-{'timestamp'})
def timing_summary(rows):return {k:{'P50_s':float(np.percentile([r.get(k,0.) for r in rows],50)),'P95_s':float(np.percentile([r.get(k,0.) for r in rows],95)),'sum_s':float(sum(r.get(k,0.) for r in rows))} for k in keys}
start=min(datetime.fromisoformat(m['started_UTC']) for m in coverage['scenes']);end=max(datetime.fromisoformat(m['completed_UTC']) for m in coverage['scenes']);bulk_start=datetime.fromisoformat(state['started_UTC'])
profile={'status':'CACHED_FULL22_COMPLETE_FINITE_LIVE_PENDING','cache_speed_label':'CACHED_GEOMETRY_AND_DETECTOR_REPLAY','NOT_RGB_TO_ID_FPS':True,
 'source_sha256':fp,'per_scene':runtime_scenes,'timings':timing_summary(all_rows),'worker_wall_s_sum':sum(m['worker_wall_s'] for m in coverage['scenes']),
 'full_prediction_elapsed_including_preflight_gap_s':(end-start).total_seconds(),'bulk_remaining21_elapsed_s':(end-bulk_start).total_seconds(),
 'four_GPU_bulk_remaining21_throughput_paired_times_per_s':(29019-1005)/(end-bulk_start).total_seconds(),
 'parallel_throughput_is_not_single_video_latency':True,'GPU_count_limit':4,'actual_GPU_indices':[5,6,7,8],
 'per_worker_peak_RSS_bytes_minmax':[min(m['peak_RSS_bytes'] for m in coverage['scenes']),max(m['peak_RSS_bytes'] for m in coverage['scenes'])],
 'per_worker_peak_CUDA_allocated_bytes_minmax':[min(m['GPU_peak_allocated_bytes'] for m in coverage['scenes']),max(m['GPU_peak_allocated_bytes'] for m in coverage['scenes'])],
 'memory_scope':'min/max of actually measured per-worker peaks; not simultaneous whole-host aggregate memory',
 'cached_rate_scope':'includes read/hash/join, head/CPU solver, numerical assertions and serialization; no RGB decode/detector/Omega forward',
 'timing_overlap':'head/evidence includes prediction; identity update includes Kalman update; nested timers must not be summed as independent components',
 'NAS_archive_status':state['archives'],'NAS_archive_in_inference_FPS':False}
live_path=root/'live_C0_C1/v4_live_runtime.json';live=None
if live_path.exists():
 live=read(live_path);assert live['status']=='COMPLETE_FINITE_TRAIN_LIVE_PREFLIGHT' and set(live['variants'])=={'C0','C1'} and live['GT_used'] is False
 assert live['checkpoint']['checkpoint_sha256']==FULL_SHA and all(v['synchronized_dual_camera_times']==183 for v in live['variants'].values())
 profile.update(status='CACHED_FULL22_AND_FINITE_REAL_LIVE_COMPLETE',finite_real_live=live,finite_RGB_to_ID_FPS={k:v['LIVE_RGB_to_final_ID_FPS'] for k,v in live['variants'].items()},
  live_scope='first61 TRAIN4/8/12 each,183 paired times per variant,same V100/equal warmup; not full-video TEST FPS')
if a.final:assert live is not None
write_json(out/'v4_c1_runtime.json',profile)
# Full per-scene native comparison: no averaging of scene HOTA into official HOTA.
per_scene=[];baseline=baselines['Full V3 + Omega']
for sid in LEGAL_TEST:
 name=scene_names[sid];c=metrics['TrackEval']['per_scene'][name];b=baseline['TrackEval']['per_scene'][name]
 row={'scene':sid,'name':name,**c,'CVIDF1':metrics['VisionTrack_MATLAB']['per_scene'][name]['CVIDF1'],'CVMA':metrics['VisionTrack_MATLAB']['per_scene'][name]['CVMA']}
 for k in ['HOTA','AssA','IDF1']:row['delta_'+k]=c[k]-b[k]
 row['delta_CVIDF1']=row['CVIDF1']-baseline['VisionTrack_MATLAB']['per_scene'][name]['CVIDF1'];per_scene.append(row)
with (out/'v4_c1_per_scene_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,list(per_scene[0]),lineterminator='\n');w.writeheader();w.writerows(per_scene)
ranked=sorted(per_scene,key=lambda r:r['delta_HOTA']);pairs=attribution['legal14_paired_C1_vs_C0']
difficulty_path=root/'v4_c1_scene_difficulty.json';difficulty=read(difficulty_path) if difficulty_path.exists() else None
if difficulty:shutil.copyfile(difficulty_path,out/'v4_c1_scene_difficulty.json')
lines=['# WorldMOT V4 C1 — 完整 TEST 正式结果','',
 '结论：C1运动状态估计相对冻结Full V3有正式净提升，但仍未超过No-Geometry、V2或GMT Native。单帧同步融合和持续身份之间仍有取舍。本轮唯一C1完整TEST已结束，不启动新训练、参数搜索或第二个TEST版本。','',
 'C1 Full TEST：**HOTA 59.3712，AssA 56.3519，IDF1 67.9896，CVIDF1 65.8169**。','',
 '所有22场完整因果推理，29,019同步时刻、58,038张原RGB对应输入、548,583个原始检测候选。检测键/BBox/Score及ROI输入差异为0；模型没有读取GT。大预测文件已逐场SHA核验归档NAS。','',
 '正式评分使用原始GT合法14场：1/2/3/4/6/7/8/9/10/11/15/16/18/19。其余5/12/13/14/17/20/21/22完整推理但只保留干净子集诊断，不加入正式成绩。TrackEval使用原生合并结果，MATLAB为原生VisionTrack R2020a，TrackEval的29个源码和MATLAB的47个原生跟踪依赖SHA与四个基线一致；未修改原始GT或原生工具。','',
 '| 方法 | HOTA | DetA | AssA | IDF1 | CVIDF1 | CVMA | MOTA | TrackEval IDSW | Frag | TP | FP | FN |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,v in values.items():lines.append('| '+name+' | '+' | '.join(f'{v[k]:.4f}' if k not in ['IDSW','Frag','CLR_TP','CLR_FP','CLR_FN'] else str(v[k]) for k in columns if k!='MATLAB_CLEAR_IDSW')+' |')
lines+=['','| C1减去基线 | ΔHOTA | ΔAssA | ΔIDF1 | ΔCVIDF1 | ΔMOTA | ΔTrackEval IDSW |','|---|---:|---:|---:|---:|---:|---:|']
for name,v in deltas.items():lines.append('| '+name+' | '+' | '.join(f'{v[k]:+.4f}' for k in ['HOTA','AssA','IDF1','CVIDF1','MOTA'])+f" | {v['IDSW']:+d} |")
lines+=['','评测版本核验：Full V3/No-Geometry与C1的驱动和文本精度相同；历史GMT/V2哈希清单另含17个未用检测工具文件。其MATLAB驱动仅少3行进度日志/partial JSON保存，移除后字节SHA精确一致。历史GMT/V2的MOT文本为.2f，当前为.17g；28份跨摄MATLAB输入GT数值矩阵与原GMT逐项完全相同。GMT/V2明确作为已发布系统基线，未宣称所有包装器SHA相同。','', '相对V3 Full，HOTA +2.9494、AssA +5.5712、IDF1 +4.0240、CVIDF1 +3.9756，TrackEval IDSW减少230。它们共享冻结权重与检测/ROI/Ω输入，差异仅为C1预测与Kalman/Joseph状态更新。','',
 '相对No-Geometry，四项主身份/跟踪指标仍低1.0605/1.9348/2.1727/1.7880；当前包含Ω的整体关联路线优于无几何仍未成立。相对GMT，HOTA/AssA/IDF1/CVIDF1分别仍差6.6933/13.2269/12.1226/12.2444。不能以MOTA较高或IDSW较少宣称超过GMT。V2的输出候选/保留规则不同，是系统参考而非单一运动模块因果对照。','',
 '输入候选完全相同仍可能有少量TrackEval TP/FP/FN/DetA差异，因为身份相关匹配及并列选择会影响评测匹配；不是检测器发生变化。','',
 '| 原生切换指标 | Full V3 | C1 | C1减Full |','|---|---:|---:|---:|',
 f"| TrackEval IDSW | {values['Full V3 + Omega']['IDSW']} | {current['IDSW']} | {deltas['Full V3 + Omega']['IDSW']:+d} |",
 f"| MATLAB interleaved CLEAR switches | {values['Full V3 + Omega']['MATLAB_CLEAR_IDSW']} | {current['MATLAB_CLEAR_IDSW']} | {deltas['Full V3 + Omega']['MATLAB_CLEAR_IDSW']:+d} |",'',
 'MATLAB CLEAR switches反而增加15，CVMA微降0.0079；与TrackEval的IDSW不是同一个数字。跨摄全局CVIDF1和帧间指标并非所有方面同步改善。','',
 '| 合法14场相同GT事件的C1对C0归因 | 事件数 | 纠正 | 回归 | 净纠正 |','|---|---:|---:|---:|---:|']
for name,c in pairs.items():lines.append(f"| {name} | {c.get('events',0)} | {c.get('corrections',0)} | {c.get('regressions',0)} | {c['net_corrections']:+d} |")
graph=attribution['legal14_identity_graph'];c0=graph['C0_Full_V3'];c1=graph['C1']
lines+=['','诊断时间连续性纠正1,089、回归876，净+213；同步同人正确共享净−68、异人正确分离净−34。全视频最佳一对一观测保留纠正17,487、回归9,996，净+7,491。这支持主要收益来自更长、更纯的主身份片段，而非提高每个时刻的同步融合准确率。','',
 f"全局最佳一对一保留：{c0['global_one_to_one_matched_clean_observations']}→{c1['global_one_to_one_matched_clean_observations']}，分母均{c0['clean_observations']}；混合ID中的少数人物观测{c0['minority_person_observations_in_mixed_ids']}→{c1['minority_person_observations_in_mixed_ids']}。但混合ID数{c0['output_ids_mixed_across_clean_people']}→{c1['output_ids_mixed_across_clean_people']}、额外身份碎片{c0['extra_global_id_fragments_per_person_summed']}→{c1['extra_global_id_fragments_per_person_summed']}，仍有代价。",'',
 'False Birth与NULL沿用原V3代理定义；共同唯一历史机会与全部当前观测分别报告，不能混淆分母。累积混合ID可能仍包含当前正确人，其复用并不全是新增异人错误。检测间隔不等于已证实物理遮挡。全局一对一保留是干净IoU匹配子集诊断，不能称为CVIDF1。','',
 '原V3干净GT匹配器、TemporalErrorAudit、历史标签更新及identity_graph均复用原代码。诊断IoU=.5匹配与正式TrackEval/原生MATLAB分开。阻塞8场单独报告：其中时间连续性净−518、同步同人净−649、异人分离净−778，不能隐藏拥挤场景退化，也不能用清洗后的这些场景改变正式分数。','',
 '| 合法14场逐场正式结果 | Scene | HOTA | AssA | IDF1 | CVIDF1 | ΔHOTA vs Full | ΔCVIDF1 vs Full |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in per_scene:lines.append(f"| {r['name']} | {r['scene']} | {r['HOTA']:.4f} | {r['AssA']:.4f} | {r['IDF1']:.4f} | {r['CVIDF1']:.4f} | {r['delta_HOTA']:+.4f} | {r['delta_CVIDF1']:+.4f} |")
lines+=['','HOTA改善最多：'+', '.join(f"Scene{r['scene']} {r['delta_HOTA']:+.4f}" for r in ranked[-3:][::-1])+'；恶化最多：'+', '.join(f"Scene{r['scene']} {r['delta_HOTA']:+.4f}" for r in ranked[:3])+'。逐场完整指标见CSV，以上排序只解释冻结结果，不决定新参数。','',
 '高密度与长视频：完整22场的逐场诊断均保留。按当前输入候选数/时刻描述拥挤度，按原始时刻数描述视频长度；非法GT场景无正式HOTA，不填伪分数。','',
 '| 完整视频难度/持续性 | GT合法 | 时刻数 | 平均候选/时刻 | 时间净纠正 | 同人净纠正 | 异人净纠正 |','|---|---|---:|---:|---:|---:|---:|']
for m in sorted(coverage['scenes'],key=lambda m:m['scene']):
 s=m['scene'];q=attribution['per_scene'][str(s)]['paired_C1_vs_C0'];lines.append(f"| Scene{s} | {s in LEGAL_TEST} | {m['timestamps']} | {m['candidate_count']/m['timestamps']:.2f} | {q['temporal_continuity']['net_corrections']:+d} | {q['same_person_shared_ID']['net_corrections']:+d} | {q['different_person_separation']['net_corrections']:+d} |")
if difficulty:
 lines+=['','| 跨视角外观分层 | GT合法 | 同人对平均ROI cosine | 低cos≤.55事件 | 纠正/回归/净 |','|---|---|---:|---:|---:|']
 for s,c in sorted(difficulty['per_scene'].items(),key=lambda item:item[1]['same_person_mean_current_raw_ROI_cosine']):
  q=c['same_person_low_ROI_cosine_C1_vs_C0'];lines.append(f"| Scene{s} | {c['original_GT_legal']} | {c['same_person_mean_current_raw_ROI_cosine']:.4f} | {q.get('events',0)} | {q.get('corrections',0)}/{q.get('regressions',0)}/{q['net_corrections']:+d} |")
 lines+=['','低ROI相似度是外观差异描述，不是标定相机角度；该固定分层的合法14场汇总逐项重现主归因统计，未选择性筛选有利GT。']
lines+=['','速度首先区分缓存与真实RGB：','',f"22场工作进程总wall之和 {profile['worker_wall_s_sum']:.3f}s；余下21场四GPU队列到最后预测交付 {profile['bulk_remaining21_elapsed_s']:.3f}s，聚合吞吐 {profile['four_GPU_bulk_remaining21_throughput_paired_times_per_s']:.4f}双摄时刻/s。这是并行缓存吞吐，不是单视频延迟，也不含RGB读取、检测或Ω前向。Scene1来自工程预检后复用，整轮首尾elapsed还包含预检到队列的间隔。",'',
 '| 缓存逐场实际耗时 | 时刻数 | worker wall(s) | cached 时刻/s | 峰值RSS MiB | 峰值CUDA MiB |','|---|---:|---:|---:|---:|---:|']
for s,v in sorted(runtime_scenes.items(),key=lambda item:int(item[0])):lines.append(f"| {s} | {v['timestamps']} | {v['worker_wall_s']:.3f} | {v['cached_paired_times_per_s']:.3f} | {v['peak_RSS_bytes']/2**20:.1f} | {v['GPU_peak_allocated_bytes']/2**20:.1f} |")
lines+=['','| 全TEST缓存阶段 | P50(s) | P95(s) | 总wall(s) |','|---|---:|---:|---:|']
for k in ['cache_read_and_join_s','head_and_evidence_transfer_s','joint_assignment_s','kalman_predict_effective_s','kalman_update_s','identity_update_s','complete_timestamp_wall_s']:
 t=profile['timings'].get(k,{'P50_s':0.,'P95_s':0.,'sum_s':0.});lines.append(f"| {k} | {t['P50_s']:.6f} | {t['P95_s']:.6f} | {t['sum_s']:.3f} |")
lines+=['','计时包含审计/输出开销；预测和Kalman为嵌套计时，不能独立相加。显存为每worker的CUDA分配峰值，RSS为进程峰值，范围不是整机同时总量。C1每时刻继续使用Ω缓存，没有跳过或减少Ω。']
if live:
 lines+=['','| 同一单V100真实RGB-to-ID有限预检 | 双摄时刻数 | 端到端FPS | Ω模型call/s | 端到端P50/P95(s) |','|---|---:|---:|---:|---:|']
 for name,v in live['variants'].items():
  t=v['timings']['RGB_to_final_ID_s'];lines.append(f"| {name} | {v['synchronized_dual_camera_times']} | {v['LIVE_RGB_to_final_ID_FPS']:.4f} | {v['Omega_model_only_per_executed_call_FPS']:.4f} | {t['P50_s']:.4f}/{t['P95_s']:.4f} |")
 lines+=['','真实流程为RGB→原GMT检测/ROI→Ω512→世界对齐→Joint Assignment→状态更新→flush ID。TRAIN4/8/12各前61时刻，每法等量5时刻热身后清空状态；只是183时刻有限预检，不能外推为完整TEST速度，亦不与历史不同场景/不同范围吞吐直接混比。所有实际Ω调用仍逐帧进行。']
else:lines+=['','正式身份评测完成后已启动同一V100的C0/C1真实有限预检。尚未完成，不用缓存速度填充RGB-to-ID。']
lines+=['','单次同卡测量不构成C1更快的统计结论；模型/对齐/系统波动均包含其中。C1的状态更新有新增工作，仍逐帧执行相同Ω调用；当前完整真实流程明显未达到实时。','', '执行和科学范围：','',
 '287项CPU测试通过。Scene1完整C0对原存档和b448af4冻结核心1005/1005时刻精确一致。CPU首次在第64时刻出现约1.4e-12中心差异，查明原R2为CUDA FP32后恢复同型号V100，未放宽比较、未调整任何Q/R或门槛。C1 Scene1输出直接复用，未重跑选择最优。','',
 'C1仅enable_motion_prediction=True；LEGACY_HARD、原Joint Assignment/Joint Birth、1024维GMT ROI、全部当前Ω支持保留；C2/C3/C4/C5未开启。6D状态/6×6协方差逐时刻验证；零dt重复predict不推进位置、不累计Q，无GT读入，场景内部不重置。原有算法文件相对44b59bc完全未修改。','',
 f'冻结Full checkpoint SHA：`{FULL_SHA}`；推理源码SHA：`{fp}`；原训练Gate SHA `363a0a687466bff7a3c8c93b2589c981ceb69a67e36fa8b58953a2d3b41aaf0d`。显式R2→V4加载器和新TEST授权锁绑定全部原始输入SHA，旧R2锁未修改/禁用。','',
 '这是用户授权的一次探索性完整TEST。设计已参考先前V3在同一TEST上的错误分析，因此不是完全未接触TEST的首次独立验证。该事实不改变当前真实正式分数，也不能被隐去。','',
 '最终判断：C1对Full V3的运动机制收益在正式TEST成立，并缩小GMT差距；但No-Geometry、V2和GMT仍更强，同步融合、局部错误与尾段污染未全面改善。当前证据不支持宣称Ω路线整体优于无几何，也不支持以MOTA/IDSW局部优势替代身份主指标。此次评测结束，不自动添加模块、调整阈值、再搜索TEST或启动六卡训练。','',
 '代码、预检、正式结果、所有基线差值、完整逐场指标、纠正与回归以及失败样例保存在本独立分支，按阶段提交。完整预测留在 `/data1/liuyeqiang/nas/A100/WorldMOT/worldobject_v4_c1_fulltest_20261026/predictions/V4_C1/`，本地保留SHA相同备份；视频、权重和大型感知缓存不上GitHub。']
lines+=['','具体正负场景的持续身份解释：','',
 'Scene7：诊断帧间变化仅63→62，HOTA却+26.9586、CVIDF1 +37.2931。冻结输出的一对一身份保留5412→9345，混合ID少数人物观测3704→31；收益主要表现为长期身份纯度和主片段归属，而不只是相邻帧少一次跳变。Scene1同样保留7638→10313，NULL代理77→23，时间变化200→166。','',
 'Scene15：时间变化480→359，但同人分裂752→779、异人误并149→170、False Birth代理16→26、错误历史复用31→44；额外碎片60→82、全局一对一保留11262→9124。这解释了少换ID仍可能HOTA下降6.4240、CVIDF1下降11.7572。','',
 'Scene16：时间变化235→296，同人分裂1500→1578、异人误并85→163，混合ID少数人物观测664→1567；一对一保留9292→8194。它同时暴露同步与时间退化，对应HOTA −5.4408、CVIDF1 −7.5075。','',
 'Scene10：时间变化246→236、误并138→133、False Birth代理14→11，但一对一保留7812→6482，少数人物污染271→388，额外碎片数仍37。局部事件数改善不能替代每个主ID的持续归属和时长；对应HOTA −4.8684、CVIDF1 −9.7464。','',
 '长视频尾段：合法14场最后四分之一的时间连续性纠正198/回归227（净−29），累积混合ID复用纠正962/回归1497（净−535）。中段的正收益没有消除尾段污染；缺少合法物理遮挡标注时，只把检测间隔称为再出现。','',
 '跨视角外观差异大的Scene19，同人当前ROI平均cosine仅0.4397；固定低cos分层2次纠正/10次回归（净−8），正式CVIDF1下降2.9513。密度最高Scene21与长视频Scene5的干净子集另报，均不进入原始合法14正式分数。','',
 '这些是冻结轨迹的观测归因。逐事件列表同时保留纠正与回归；仅凭统计不将全部退化归为某个Kalman噪声或空间门控参数，本轮不据此改模型或重跑TEST。']
(docs/'V4_C1_FULL_TEST_FINAL.md').write_text('\n'.join(lines)+'\n')
status={'status':'COMPLETE_FULL_TEST_NATIVE_AND_ATTRIBUTION'+('_AND_FINITE_LIVE' if live else '_FINITE_LIVE_RUNNING'),'variant':'C1','source_sha256':fp,'checkpoint_sha256':FULL_SHA,'CPU_tests':287,
 'C0_scene1_exact_frames':1005,'inference_scenes':22,'paired_timestamps':29019,'candidate_count':548583,'candidate_differences':0,'ROI_mutations':0,
 'official_raw_GT_legal_scenes':list(LEGAL_TEST),'TrackEval_status':'COMPLETE','MATLAB_status':'COMPLETE','MATLAB_version':metrics['VisionTrack_MATLAB']['MATLAB_version'],
 'HOTA':current['HOTA'],'AssA':current['AssA'],'IDF1':current['IDF1'],'CVIDF1':current['CVIDF1'],'CVMA':current['CVMA'],'MOTA':current['MOTA'],'TrackEval_IDSW':current['IDSW'],
 'identity_attribution_status':'COMPLETE','NAS_archive_scenes_verified':22,'inference_no_GT':True,'exploratory_TEST_not_first_blind_holdout':True,
 'beats_Full_V3_primary_metrics':True,'beats_No_Geometry_primary_metrics':False,'beats_GMT_primary_metrics':False,'new_training_started':False,'TEST_parameters_changed':False,
 'additional_TEST_search_started':False,'finite_real_LIVE_status':'COMPLETE' if live else 'RUNNING','created_UTC':datetime.now(timezone.utc).isoformat()}
write_json(out/'v4_c1_final_status.json',status)
write_json(out/'v4_c1_report_provenance.json',{'status':'VERIFIED_MEASURED_REPORT','source_sha256':fp,'builder_sha256':sha256(__file__),'raw_native_metrics_sha256':sha256(root/'evaluation/V4_C1/metrics.json'),'raw_identity_attribution_sha256':sha256(root/'v4_c1_identity_attribution.json'),'raw_coverage_sha256':sha256(root/'v4_c1_fulltest_coverage.json'),'shared_native_tracking_dependency_hashes_match':True,'baseline_tool_comparability':tool_comparability,'baseline_proofs':base_proofs,'generated_UTC':datetime.now(timezone.utc).isoformat()})
print(json.dumps({'official':{k:current[k] for k in ['HOTA','AssA','IDF1','CVIDF1','CVMA','MOTA','IDSW']},'Full_delta':deltas['Full V3 + Omega'],'LIVE_complete':live is not None,'difficulty_complete':difficulty is not None,'worst_scenes':[(x['scene'],x['delta_HOTA']) for x in ranked[:3]],'best_scenes':[(x['scene'],x['delta_HOTA']) for x in ranked[-3:]]}))
