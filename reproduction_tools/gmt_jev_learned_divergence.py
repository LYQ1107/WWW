"""Locate earliest native-ID discrepancy; never treat later numeric IDs as errors."""
import collections,gzip
from gmt_jev_common import *

def load(reference):
 assert sha(reference['path'])==reference['SHA256']
 with gzip.open(reference['path'],'rt') as f:return json.load(f)

def main():
 baseline_ref=read(REPORTS/'ORIGINAL_GMT_METRICS.json')['predictions'];base=load(baseline_ref)
 images={i['id']:i for i in read(ANNOTATIONS)['images'] if i['video_id']==16}
 first_scene=[p for p in base if images[p['image_id']]['frame_id']==1]
 counts=collections.Counter(images[p['image_id']]['view_id'] for p in first_scene)
 assert len(counts)==2
 trace_ref=read(REPORTS/'BASELINE_PARITY.json')['cases'][0]
 assert sha(trace_ref['path'])==trace_ref['SHA256']
 first_commit=read(trace_ref['path'])['commits'][0]['key']
 assert first_commit[:2]==[16,0] and first_commit[2] in [0,1]
 # In two cameras, the first association commit is the non-bootstrap camera.
 bootstrap_view=2-first_commit[2];methods={}
 for kind in ['mlp','attention','cic']:
  source=read(REPORTS/f'ONLINE_{kind}_learned.json');reference=source['raw_predictions'];pred=load(reference)
  assert len(pred)==len(base);different=[];first_scene_changes=collections.Counter();row_index=collections.Counter()
  for a,b in zip(base,pred):
   assert {k:v for k,v in a.items() if k!='track_id'}=={k:v for k,v in b.items() if k!='track_id'}
   image=images[a['image_id']];f,v=image['frame_id']-1,image['view_id']-1;row=row_index[a['image_id']];row_index[a['image_id']]+=1
   if a['track_id']!=b['track_id']:
    different.append(dict(frame=f,view=v,row=row,original_ID=a['track_id'],modified_ID=b['track_id']))
    if f==0:first_scene_changes[v]+=1
  first=min(different,key=lambda x:(x['frame'],x['view'],x['row'])) if different else None
  methods[kind]=dict(predictions=reference,trained_checkpoint=source['trained_checkpoint'],
   earliest_native_numeric_discrepancy=first,
   first_scene_changed_numeric_rows_by_view=dict(first_scene_changes),
   first_scene_examples=[x for x in different if x['frame']==0][:10],
   bootstrap_view_unchanged=not first_scene_changes[bootstrap_view-1])
 result=dict(status='COMPLETE_EARLIEST_NATIVE_OUTPUT_DISCREPANCY_AUDIT',video=16,baseline=baseline_ref,
  metadata_view_ids_are_one_based=True,reported_event_views_are_zero_based=True,
  first_scene_detection_counts_by_view=dict(counts),native_bootstrap_view=bootstrap_view,methods=methods,
  initial_unique_detection_count_assumption_rejected=len(set(counts.values()))==1,
  bootstrap_evidence=dict(original_native_trace=trace_ref,first_association_commit=first_commit,
   rule='Two-camera native bootstrap has no association-commit callback; first retained association commit is the other camera.'),
  first_event_order='Lexicographic frame/view/row; within-scene order is not an execution-time ordering.',
  no_new_GMT_run=True,no_gradient_updates=True,GT_labels_used=False,source=binding(),
  interpretation='All runs start cold with the same seed, frontend and native rules. Bootstrap camera is resolved using the original native commit trace even when detection counts tie. Numeric differences at the first affected scene locate a possible intervention. Later numeric IDs can shift with allocation and are never counted as semantic errors here.',
  causal_attribution='NOT_ESTABLISHED: no first-only/no-first paired intervention executed; earliest discrepancy alone does not explain HOTA or actual error propagation.')
 save(REPORTS/'LEARNED_EARLIEST_DIVERGENCE.json',result)
 lines=['# 起始原生输出分歧审计','',
  '只复用已保留预测、原生提交记录和图像元数据，不使用GT标签、不跑新模型、不调整权重。所有raw几何/检测置信度逐项一致。',
  '第一场景两相机各30个检测，初始化相机由原始native trace的首次关联提交（另一个相机）确认；不假设检测数唯一。事件中的frame/view均从0开始，同场景view排序不是执行时间排序。','',
  '| 方法 | 最早分歧 frame/view/row | 第一场景各view数字ID不同的行数 | bootstrap相机保持 |','|---|---|---|---|']
 for kind,r in methods.items():
  e=r['earliest_native_numeric_discrepancy'];key=str([e['frame'],e['view'],e['row']]) if e else 'NONE'
  lines.append(f"| {kind} | {key} | {r['first_scene_changed_numeric_rows_by_view']} | {r['bootstrap_view_unchanged']} |")
 lines+=['','最早分歧能定位初始干预的候选时点，不能说明这就是收益原因。后续ID数字受分配器和状态分歧影响，不能据数字差异统计身份错误。完整质量仍用冻结评测器。',
  '若收益起于首帧，需要first-only/no-first的同起始状态完整干预才能归因；本审计没有执行该消融，不把它称作纠错机制或真实传播时间证明。']
 (DOCS/'LEARNED_EARLIEST_DIVERGENCE.md').write_text('\n'.join(lines)+'\n')
 print('EARLIEST_NATIVE_DISCREPANCY',{k:r['earliest_native_numeric_discrepancy'] for k,r in methods.items()},flush=True)

if __name__=='__main__':main()
