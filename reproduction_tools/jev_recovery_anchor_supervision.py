"""Past-certified native-ID anchors: explicit persistence target, never future safety.

An ID's first pure observed certificate defines this *new* frozen persistence
target. Later mixed history does not retroactively erase that real certificate.
Unbound/unknown current identities remain UNKNOWN. Old purity labels are intact.
"""
import collections,gc,gzip,torch,traceback
from jev_recovery_common import *
from jev_phase14_artifacts import load_dense,save_dense
from jev_oneweek_supervision import certify
from jev_oneweek_learning import comparisons
def main():
 protect();torch.set_num_threads(1)
 protocol=read(REPORTS/'R2_ANCHOR_LABEL_PROTOCOL.json')
 assert protocol['status']=='FROZEN_BEFORE_ANCHOR_COUNTS'
 import jev_oneweek_common as old
 from jev_phase13_runtime import cache_inputs
 from build_dense_jev_stage2_dataset import OfflineLabels
 labels={tuple(r['key']):r for r in load_dense(read(REPORTS/'R2_QUESTION_SUPERVISION.json')['labels']['path'])['labels']}
 proposals={tuple(r['key']):r for r in read(OUT/'audit/PROPOSAL_PATCH.json')['records']}
 patches=[];summaries=[]
 for video in TRAIN:
  folder=old.PHASE15/'commitment_dataset_v2'/f'video{video:02d}';source=read(folder/'RESULT.json')['DATASET'];assert sha(source['path'])==source['SHA256']
  records=load_dense(source['path'])['records'];lookup={tuple(r['key'][1:]):r for r in records}
  _,_,reader=cache_inputs(video);offline=OfflineLabels(video,reader)
  for key,image in offline.images.items():
   c=collections.Counter(a['instance_id'] for a in offline.gt[image['id']]);bad={g for g,n in c.items() if n>1}
   if bad:offline.aligned[key]=[None if g in bad else g for g in offline.current(*key)]
  with gzip.open(folder/'COMMITS.jsonl.gz','rt') as f:commits=[json.loads(line) for line in f]
  votes=collections.defaultdict(collections.Counter);total=collections.Counter();anchor={};cert_keys={};count=collections.Counter();groups=collections.defaultdict(set)
  def update(frame,view,ids,gt):
   assert len(ids)==len(gt)
   for identity,g in zip(ids,gt):
    total[identity]+=1
    if g is not None:votes[identity][g]+=1
    if identity not in anchor:
     owner=certify(votes[identity],total[identity])
     if owner is not None:anchor[identity]=owner;cert_keys[identity]=(frame,view)
  boot=1-int(commits[0]['key'][1]);gt=offline.current(0,boot);update(0,boot,list(range(1,len(gt)+1)),gt)
  for event in commits:
   frame,view=map(int,event['key']);r=lookup.get((frame,view));gt=offline.current(frame,view)
   if r is not None:
    assert gt==r['GT_labels_OFFLINE_ONLY'];q,k=r['inputs']['legal'][0].shape
    known=torch.zeros(q,k+1,dtype=torch.bool);positive=known.clone()
    for row,g in enumerate(gt):
     if g is None:continue
     for col,identity in enumerate(r['refs']):
      if identity in anchor and r['inputs']['legal'][0,row,col]:
       assert cert_keys[identity]<(frame,view) or cert_keys[identity][0]==frame and cert_keys[identity][1]!=view
       known[row,col]=True;positive[row,col]=anchor[identity]==g
       if not r['known_options'][row,col]:count['newly_certified_persistence_edges']+=1
     # A private MATCH DEFER is wrong when an authenticated compatible active ID
     # is available. It is valid only when all legal owners are certified absent.
     # Otherwise keep UNKNOWN, including ambiguous unbound active IDs.
     has_correct=bool(positive[row,:k].any());all_certified=bool((known[row,:k]|~r['inputs']['legal'][0,row]).all())
     known[row,k]=has_correct or all_certified;positive[row,k]=not has_correct and all_certified
     count['known_current_rows']+=1;count['compatible_active_anchor_rows']+=has_correct
    old_known=r['known_options'][:,:k]&r['inputs']['legal'][0]
    assert torch.equal(positive[:,:k][old_known],r['positive'][:,:k][old_known]),'Previously pure known owners may not contradict persistent anchors'
    annotation=labels[tuple(r['key'])];record=dict(r,known_options=known,positive=positive,oneweek_labels=annotation)
    meta=dict(key=r['key'],known_options=known,positive=positive,certificate='First actual past pure >=3 known/80%-covered observed GT per native ID; target is retention of this authenticated anchor, not true birth owner or future-safe action')
    pp=proposals[tuple(r['key'])]['proposals']['P1_8']['plans'];pairs=comparisons(dict(plans=[pp]),None,[record]);count['certified_local_rank_pairs']+=len(pairs);count['events_with_local_rank']+=bool(pairs)
    for _,a,b,sign,n in pairs:
     changed=[rr for rr,(u,v) in enumerate(zip(pp[a],pp[b])) if u!=v]
     people=tuple(sorted({gt[rr] for rr in changed if gt[rr] is not None}));ids=tuple(sorted({r['refs'][c] for p in (pp[a],pp[b]) for rr,c in enumerate(p) if rr in changed and c>=0}))
     stratum='gradient' if annotation['train_gradient'] else 'reserved' if video!=16 else 'exposed_audit'
     groups[stratum].add((video,people,ids,frame//64,view))
    patches.append(meta)
   update(frame,view,event['ids'],gt)
  summaries.append(dict(video=video,counts=dict(count),certified_conflict_groups={s:len(v) for s,v in groups.items()},certified_native_IDs=len(anchor),source=source))
  save(OUT/'anchor_supervision/PROGRESS.json',dict(status='RUNNING',complete_videos=summaries));print('ANCHOR_SUPERVISION_VIDEO',video,dict(count),{s:len(v) for s,v in groups.items()},flush=True)
  del records,lookup,offline,reader,commits;gc.collect()
 p=OUT/'anchor_supervision/PATCH.pth.xz';p.parent.mkdir(exist_ok=True);save_dense(p,dict(patches=patches,protocol=ref(REPORTS/'R2_ANCHOR_LABEL_PROTOCOL.json')),reserve=30*2**30)
 eligible=any(v['certified_conflict_groups'].get('gradient',0)>0 for v in summaries)
 save(REPORTS/'R2_ANCHOR_SUPERVISION.json',dict(status='COMPLETE',binding=binding('Past-authenticated ID persistence target audit'),videos=summaries,patch=ref(p),current_persistence_supervision_eligible=eligible,old_current_purity_labels_unchanged=True,old_future_safety_gate_unchanged=True,true_birth_owner_not_claimed=True,current_target_convention='Preserve first past-certified native ID/person binding',inference_GT=False,TRAIN16_gradient=False,independent_validation=False))
 print('ANCHOR_SUPERVISION_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:failure('anchor_supervision',traceback.format_exc());raise
