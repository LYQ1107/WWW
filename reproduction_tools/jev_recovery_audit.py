"""All authorized TRAIN events: dependence-aware targets and GT-free legal coverage.

GT is used only after proposals/scores exist. Historical bank membership is not
saved in this frozen corpus; past compatible absent IDs are *not* called Bank.
"""
import collections,gc,gzip,torch
from jev_recovery_common import *
from jev_phase14_artifacts import load_dense
from jev_oneweek_learning import batch,comparisons
from jev_oneweek_assess import load_model
from gtr.modeling.jev_recovery.plans import plans

def main():
 protect();torch.set_num_threads(1)
 import jev_oneweek_common as old
 source=read(ROOT/'reports/JEV_ONEWEEK/SUPERVISION_AUDIT.json')
 assert sha(source['labels']['path'])==source['labels']['SHA256']
 labels={tuple(r['key']):r for r in load_dense(source['labels']['path'])['labels']}
 actor,checkpoint,_=load_model('jev',20261009,'formal');actor.eval()
 census=[];cover=[];effective=[];patch=[];global_groups=collections.defaultdict(set)
 with torch.no_grad():
  for video in TRAIN:
   folder=old.PHASE15/'commitment_dataset_v2'/f'video{video:02d}'
   manifest=read(folder/'RESULT.json');descriptor=manifest['DATASET'];assert sha(descriptor['path'])==descriptor['SHA256']
   records=load_dense(descriptor['path'])['records'];count=collections.Counter();coverage=collections.defaultdict(collections.Counter)
   groups=collections.defaultdict(set);strata=collections.defaultdict(collections.Counter);rank_groups=collections.defaultdict(set);rank_count=collections.Counter()
   for index,r in enumerate(records):
    annotation=labels[tuple(r['key'])];a=annotation['cross'];b=annotation['stability'];legal=r['inputs']['legal'][0];q,k=legal.shape
    frame,view=r['key'][1:];stratum='gradient' if annotation['train_gradient'] else 'reserved' if video!=16 else 'exposed_audit'
    x,y=batch([dict(r,oneweek_labels=annotation)]);d=actor.details(x);z=d['base_logits'][0,:q,:k+1]
    proposed={name:plans(z,legal.to(z.device),maximum=n,mode=m) for name,n,m in [('P0_4',4,'old'),('P1_4',4,'expanded'),('P1_8',8,'expanded')]}
    def add(pattern,row,col):
     count[pattern]+=1;strata[stratum][pattern]+=1
     gt=r['GT_labels_OFFLINE_ONLY'][row];ref_id=r['refs'][col]
     tags=dict(video=(video,),person=(video,gt),camera_pair=(video,view,1-view),
       Global_ID_conflict=(video,gt,ref_id),temporal_block=(video,gt,view,frame//64),
       history_dependency=(video,ref_id))
     for level,key in tags.items():groups[pattern,level].add(key);global_groups[pattern,level].add(key)
    for row in range(q):
     for col in torch.where(legal[row])[0].tolist():
      aa,bb=int(a[row,col]),int(b[row,col])
      name='both_UNKNOWN' if aa<0 and bb<0 else 'one_UNKNOWN' if aa<0 or bb<0 else f'Cross{aa}_Continuity{bb}'
      add(name,row,col)
      if aa>=0 and bb>=0 and aa!=bb:add('target_disagreement',row,col)
     positive=r['positive'][row,:k]&legal[row];known=r['known_options'][row,:k]&legal[row]
     if bool(positive.any()) and bool((known&~positive).any()):count['legal_certified_correct_and_wrong_competition_rows']+=1
     if int(positive.sum())>1:count['same_person_multiple_certified_native_ID_alias_rows']+=1
     if r['GT_labels_OFFLINE_ONLY'][row] is None:count['current_GT_UNKNOWN_rows']+=1
     elif not bool(positive.any()):count['no_certified_compatible_active_ID_rows']+=1
     for name,pp in proposed.items():
      c=coverage[name];c['rows']+=1;c['legal_active_compatible_rows']+=bool(positive.any())
      contained=any(p[row]>=0 and bool(positive[p[row]]) for p in pp)
      if bool(positive.any()):
       c['compatible_active_and_proposed_rows']+=contained;c['B_certified_correct_active_but_not_proposed']+=not contained
      else:c['A_or_E_no_certified_correct_active_ID']+=1
      chosen=int(actor.score_plans(d,x,0,pp).argmax());selected=pp[chosen][row];col=k if selected<0 else selected
      if contained and bool(r['known_options'][row,col]):c['C_covered_but_chosen_known_wrong']+=not bool(r['positive'][row,col])
      elif contained:c['E_covered_but_chosen_UNKNOWN']+=1
    record_meta=dict(key=r['key'],train_gradient=annotation['train_gradient'],proposals={})
    for name,pp in proposed.items():
     dd=dict(plans=[pp]);pairs=comparisons(dd,y,[dict(r,oneweek_labels=annotation)])
     rank_count[name+'_events']+=1;rank_count[name+'_certified_rank_pairs']+=len(pairs);rank_count[name+'_events_with_pairs']+=bool(pairs)
     keys=[]
     for _,ai,bi,sign,changed in pairs:
      rows=[rr for rr,(u,v) in enumerate(zip(pp[ai],pp[bi])) if u!=v]
      people=tuple(sorted({r['GT_labels_OFFLINE_ONLY'][rr] for rr in rows if r['GT_labels_OFFLINE_ONLY'][rr] is not None}))
      identities=tuple(sorted({r['refs'][c] for p in [pp[ai],pp[bi]] for rr,c in enumerate(p) if rr in rows and c>=0}))
      group=(video,people,identities,frame//64,view);rank_groups[name,stratum].add(group);keys.append(group)
     record_meta['proposals'][name]=dict(plans=[list(p) for p in pp],rank_pairs=len(pairs),independent_group_keys=keys)
    patch.append(record_meta)
    if index%128==0:
     save(OUT/'audit/PROGRESS.json',dict(status='RUNNING',video=video,record=index,total=len(records),complete_videos=[c['video'] for c in census]))
     print('RECOVERY_AUDIT_PROGRESS',video,index,len(records),flush=True)
   census.append(dict(video=video,records=len(records),counts=dict(count),strata={s:dict(c) for s,c in strata.items()},group_counts=[dict(pattern=p,level=l,groups=len(v)) for (p,l),v in groups.items()],source=ref(folder/'RESULT.json')))
   cover.append(dict(video=video,variants={n:dict(c) for n,c in coverage.items()}))
   effective.append(dict(video=video,counts=dict(rank_count),certified_local_conflict_groups=[dict(variant=n,stratum=s,groups=len(v)) for (n,s),v in rank_groups.items()]))
   del records,r,x,y,d;gc.collect()
 save(OUT/'audit/PROPOSAL_PATCH.json',dict(records=patch))
 save(REPORTS/'QUESTION_DISAGREEMENT_CENSUS.json',dict(status='COMPLETE',binding=binding('All authorized TRAIN predicted-state events'),videos=census,total_group_counts=[dict(pattern=p,level=l,groups=len(v)) for (p,l),v in global_groups.items()],label_definition='Cross1/Continuity1 means independently certified targets, not classifier correctness',history_dependency='All events sharing the actual native ID are one conservative cluster per video; not IID edges',all_TRAIN_scope=TRAIN,other_TRAIN_not_authorized_for_new_gradients=True))
 save(REPORTS/'QUESTION_EFFECTIVE_SUPERVISION.json',dict(status='COMPLETE',videos=effective,proposal_patch=ref(OUT/'audit/PROPOSAL_PATCH.json'),checkpoint=checkpoint,changed_UNKNOWN_disqualifies_ranking=True,all_capacity_affected_changed_rows_checked=True,unchanged_UNKNOWN_cancel=True,future_safety_qualified=False))
 save(REPORTS/'LEGAL_ASSOCIATION_COVERAGE.json',dict(status='COMPLETE',videos=cover,checkpoint=checkpoint,inference_GT=False,original_cold_trajectory_reused_for_diagnostic_only=True,complete_new_online_still_required=True,D_bank_membership='UNKNOWN: frozen dense records omit actual Bank membership. Absent historical IDs are not certified recoverable Bank IDs.',A_E_distinction='No certified compatible active ID includes either physical absence or contaminated/UNKNOWN owner; never label UNKNOWN wrong',F_aliases='All certified same-person native IDs counted compatible; no claim numeric ID restoration',cost='Bounded <=32 constrained solves per expanded payload plus original proposals; full real-image FPS not yet measured'))
 (DOCS/'QUESTION_LABEL_CONFLICTS.md').write_text('# 问题标签与依赖\n\n沿用冻结标签，不制造分歧。Cross 是其他摄像头过去纯身份与当前人一致；Continuity 是近期本摄像头已认证 ID 的延续。混合历史保持 UNKNOWN，当前可兼容的同人别名允许 Cross=1、Continuity=0。相邻边按人、实际 Global ID、64 帧块去重；共享 ID 历史依赖组不能当 IID 样本。零分歧审计视频不能证明两问题独立有效。\n')
 (DOCS/'QUESTION_DISAGREEMENT_ANALYSIS.md').write_text('# 双问题有效监督审计\n\n完整计数与独立组数见 QUESTION_DISAGREEMENT_CENSUS.json 和 QUESTION_EFFECTIVE_SUPERVISION.json。统计范围严格为已授权 TRAIN12/13/14/16；TRAIN16 不训练。保留每视频和暴露层次；不以候选边总数代替独立冲突数量。\n')
 (DOCS/'LEGAL_ASSOCIATION_COVERAGE.md').write_text('# 合法关联可达性\n\n比较 P0 四方案、P1 四方案、P1 八方案。GT 仅在生成后标记已认证正确 active ID 是否可达，所有方法获得同样 GT-free 方案规则。冻结数据未保存真实 Bank 成员，D 类不可认证，不能将过去但不在 active 的 ID 冒充 Bank 可恢复 ID。污染 owner 无法认证时不能区分物理缺失和 UNKNOWN。扩展四方案预注册用于第一版，八方案只诊断；不得根据 DEV 选择。\n')
 print('RECOVERY_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;failure('census_coverage',traceback.format_exc());raise
