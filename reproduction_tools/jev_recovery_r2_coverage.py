"""Freeze actual early-evidence proposals and new persistent-question conflict groups."""
import collections,gc,torch,traceback
from jev_recovery_common import *
from jev_recovery_learning import load_records,initialize,batch,comparisons
def main():
 protect();torch.set_num_threads(1);train,validation,sources=load_records(2)
 # Include old reserved blocks for diagnosis, without adding them to gradients.
 from jev_oneweek_common import PHASE15
 from jev_phase14_artifacts import load_dense
 labels={tuple(r['key']):r for r in load_dense(read(REPORTS/'R2_QUESTION_SUPERVISION.json')['labels']['path'])['labels']}
 anchors={tuple(r['key']):r for r in load_dense(read(REPORTS/'R2_ANCHOR_SUPERVISION.json')['patch']['path'])['patches']}
 early={tuple(r['key']):r for r in load_dense(read(REPORTS/'R2_EARLY_EVIDENCE.json')['patch']['path'])['patches']}
 model,_=initialize('jev',20261009,2);model.eval();summaries=[];patch=[];question=[]
 with torch.no_grad():
  for video in TRAIN:
   records=load_dense(read(PHASE15/'commitment_dataset_v2'/f'video{video:02d}/RESULT.json')['DATASET']['path'])['records'];c=collections.Counter();qc=collections.Counter();groups=collections.defaultdict(set);qgroups=collections.defaultdict(set)
   for idx,r in enumerate(records):
    key=tuple(r['key']);meta=anchors[key];ev=early[key];k=len(r['refs']);q=len(r['rows'])
    target=torch.full_like(labels[key]['stability'],-1);known=meta['known_options'][:,:k];target[known]=meta['positive'][:,:k][known].to(torch.int8)
    annotation=dict(labels[key],stability=target);record=dict(r,known_options=meta['known_options'],positive=meta['positive'],oneweek_labels=annotation,anchor_visual=ev['anchor_visual'],anchor_mask=ev['anchor_mask'])
    x,y=batch([record]);d=model.details(x);pp=d['plans'][0]
    from gtr.modeling.jev_oneweek.plans import plans as original_plans
    p0=original_plans(d['base_logits'][0],x['legal'][0]);pos=meta['positive'][:,:k]&r['inputs']['legal'][0]
    c['payloads']+=1;c['rows']+=q;c['correct_active_anchor_rows']+=int(pos.any(-1).sum())
    for row in range(q):
     a=any(p[row]>=0 and bool(pos[row,p[row]]) for p in p0);b=any(p[row]>=0 and bool(pos[row,p[row]]) for p in pp)
     c['P0_correct_proposal_rows']+=a;c['P2_correct_proposal_rows']+=b;c['newly_covered_rows']+=b and not a;c['lost_covered_rows']+=a and not b
    assert c['lost_covered_rows']==0
    pairs=comparisons(dict(plans=[pp]),y,[record]);c['rank_pairs']+=len(pairs);c['rank_events']+=bool(pairs);local=[]
    stratum='gradient' if annotation['train_gradient'] else 'reserved' if video!=16 else 'exposed_audit'
    for _,a,b,sign,n in pairs:
     changed=[row for row,(u,v) in enumerate(zip(pp[a],pp[b])) if u!=v]
     people=tuple(sorted({r['GT_labels_OFFLINE_ONLY'][row] for row in changed if r['GT_labels_OFFLINE_ONLY'][row] is not None}));ids=tuple(sorted({r['refs'][col] for p in (pp[a],pp[b]) for row,col in enumerate(p) if row in changed and col>=0}))
     g=(video,people,ids,r['key'][1]//64,r['key'][2]);local.append(g);groups[stratum].add(g)
    patch.append(dict(key=key,train_gradient=annotation['train_gradient'],plans=[list(p) for p in pp],independent_group_keys=local))
    a=annotation['cross'];b=annotation['stability'];both=(a>=0)&(b>=0);different=both&(a!=b)
    qc['joint_known_edges']+=int(both.sum());qc['disagreement_edges']+=int(different.sum())
    for row,col in torch.nonzero(different).tolist():
     gt=r['GT_labels_OFFLINE_ONLY'][row];ref_id=r['refs'][col];qgroups['person'].add((video,gt));qgroups['ID_history'].add((video,ref_id));qgroups['person_ID_block'].add((video,gt,ref_id,r['key'][1]//64,r['key'][2]))
    if idx%128==0:
     save(OUT/'R2_coverage/PROGRESS.json',dict(status='RUNNING',video=video,record=idx,total=len(records),counts=dict(c)));print('R2_COVERAGE',video,idx,len(records),flush=True)
   summaries.append(dict(video=video,counts=dict(c),certified_conflict_groups={s:len(g) for s,g in groups.items()}));question.append(dict(video=video,counts=dict(qc),groups={s:len(g) for s,g in qgroups.items()}))
   del records;gc.collect()
 save(OUT/'R2_coverage/PROPOSAL_PATCH.json',dict(records=patch))
 save(REPORTS/'R2_CANDIDATE_AND_QUESTION_AUDIT.json',dict(status='COMPLETE',binding=binding('Frozen R2 GT-free early proposals and distinct persistent targets'),candidate_coverage=summaries,question_disagreement=question,patch=ref(OUT/'R2_coverage/PROPOSAL_PATCH.json'),old_purity_results_unchanged=True,coverage_gain_is_not_JEV_architecture_gain=True,early_evidence_offered_to_all_controls=True,labels_not_future_safety=True))
 print('R2_COVERAGE_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:failure('R2_candidate_supervision_audit',traceback.format_exc());raise
