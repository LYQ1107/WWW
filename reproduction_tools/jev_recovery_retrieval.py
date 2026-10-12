"""Common observation/query denominator; causal native-ID lookups, event/person macros."""
import collections,gc
from jev_recovery_common import *
def aligned(arm,seed,video):
 from jev_phase7_offline import IdentityEvaluator
 path=OLD_OUT/'online_v1/formal'/arm/f'seed{seed}'/f'video{video:02d}/RESULT.json'
 result=read(path);p=result['raw_predictions'];assert sha(p['path'])==p['SHA256']
 evaluator=IdentityEvaluator(video);rows=evaluator.align(read(p['path']));bad={}
 for image in evaluator.images.values():
  c=collections.Counter(a['instance_id'] for a in evaluator.gt[image['id']]);bad[image['frame_id']-1,image['view_id']-1]={g for g,n in c.items() if n>1 or g<0}
 for (f,v,r),value in rows.items():
  if value['gt'] in bad[f,v]:value['gt']=None
 return rows,ref(path)
def summary(c):
 tp,fp=c['TP'],c['FP'];den=c['queries'];return dict(counts=dict(c),pair_precision=tp/(tp+fp) if tp+fp else None,pair_recall=tp/c['relevant'] if c['relevant'] else None,event_precision_answered=c['sum_event_precision']/c['answered'] if c['answered'] else None,event_precision_empty_zero=c['sum_event_precision']/den if den else None,event_recall=c['sum_event_recall']/den if den else None,query_answer_coverage=c['answered']/den if den else None,mean_returned_known_history=c['known_returns']/den if den else None,mean_relevant_history=c['relevant']/den if den else None)
def main():
 protect();arms=['jev','threshold','mlp','attention','XIII_fixed_reference','parent'];cases=[];sources=[]
 for seed in SEEDS:
  for video in DEV:
   runs={}
   for arm in arms:runs[arm],s=aligned(arm,seed,video);sources.append(s)
   base=runs['parent'];assert all(set(r)==set(base) for r in runs.values())
   for r in runs.values():assert all(r[k]['gt']==base[k]['gt'] for k in base)
   groups=collections.defaultdict(list)
   for key in base:groups[key[:2]].append(key)
   boot=max((0,1),key=lambda v:(len(groups[0,v]),v));order=sorted(groups,key=lambda p:(p[0],0 if p[0]==0 and p[1]==boot else 1 if p[0]==0 else p[1]))
   indexes={a:collections.defaultdict(collections.Counter) for a in arms};prior=collections.Counter();counts={a:collections.Counter() for a in arms};persons={a:collections.defaultdict(collections.Counter) for a in arms};pairs={a:collections.defaultdict(collections.Counter) for a in arms}
   for frame,view in order:
    keys=sorted(groups[frame,view])
    for key in keys:
     gt=base[key]['gt']
     if gt is None or not prior[gt,1-view]:continue
     relevant=prior[gt,1-view]
     for arm in arms:
      returned=indexes[arm][runs[arm][key]['id'],1-view];tp=returned[gt];fp=sum(n for g,n in returned.items() if g is not None and g!=gt);unknown=returned[None];known=tp+fp
      delta=dict(queries=1,TP=tp,FP=fp,UNKNOWN_returns=unknown,relevant=relevant,known_returns=known,answered=int(known>0),empty_known_returns=int(known==0),no_relevant_return=int(tp==0),sum_event_precision=tp/known if known else 0.,sum_event_recall=tp/relevant,polluted_queries=int(fp>0))
      for c in (counts[arm],persons[arm][gt],pairs[arm][view,1-view]):c.update(delta)
    for key in keys:
     gt=base[key]['gt']
     if gt is not None:prior[gt,view]+=1
     for arm in arms:indexes[arm][runs[arm][key]['id'],view][gt]+=1
   assert len({c['queries'] for c in counts.values()})==1
   for arm in arms:
    person=[dict(person=g,**summary(c)) for g,c in sorted(persons[arm].items())]
    cases.append(dict(arm=arm,seed=seed,video=video,**summary(counts[arm]),persons=person,camera_pairs=[dict(pair=list(p),**summary(c)) for p,c in pairs[arm].items()],person_macro_event_precision_empty_zero=sum(p['event_precision_empty_zero'] for p in person)/len(person) if person else None,person_macro_event_recall=sum(p['event_recall'] for p in person)/len(person) if person else None))
   save(OUT/'retrieval/PROGRESS.json',dict(status='RUNNING',seed=seed,video=video,complete_cases=len(cases)));print('RECOVERY_FIXED_QUERY_RETRIEVAL',seed,video,flush=True)
   del runs,base;gc.collect()
 result=dict(status='COMPLETE',binding=binding('Frozen historical formal runs with exact common current/previous observation query denominator'),cases=cases,sources=sources,query_rule='Every identical current known detection with at least one earlier known same-person observation in other camera; all policies same denominator. Query before own camera payload commits.',no_GT_ID_mapping=True,actor_GT=False,independence='Person/video/history trajectories and seeds dependent; millions of returns are not independent corrections',empty_precision_convention='Answer-only event precision and fixed-denominator event precision with empty=0 both shown; pair precision separately',scope='Old trajectories diagnostic, no retraining or model selection from DEV')
 save(REPORTS/'CROSS_CAMERA_ADVANTAGE_AUDIT.json',result);save(REPORTS/'CROSS_CAMERA_RETRIEVAL.json',result)
 (DOCS/'CROSS_CAMERA_IDENTITY_ANALYSIS.md').write_text('# 固定查询分母检索\n\n相同感知观测、相同已知当前人物、相同过去目标观测提供查询。Global ID 返回来自各策略真实过去索引，无整视频最优 GT-ID 映射。分别报告空返回、已知回答覆盖、每次查询平均和每人平均，避免长轨迹返回对数放大。UNKNOWN 返回单独统计。此处是旧 DEV 诊断，不作参数选择或独立验证。\n')
 print('RECOVERY_RETRIEVAL_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;failure('fixed_query_retrieval',traceback.format_exc());raise
