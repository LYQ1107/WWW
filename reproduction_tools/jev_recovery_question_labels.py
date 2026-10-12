"""R2: cross-view support vs observed concrete-ID continuity, actual past evidence only.

Known historical mixtures certify a reliability violation; they do not identify
the correct owner. Association known_options remain UNKNOWN and unchanged.
"""
import collections,gc,gzip,torch,traceback
from jev_recovery_common import *
from jev_phase14_artifacts import load_dense,save_dense
from jev_oneweek_supervision import certify
def main():
 protect();torch.set_num_threads(1)
 protocol=read(REPORTS/'R2_LABEL_PROTOCOL.json');assert protocol['status']=='FROZEN_BEFORE_R2_LABEL_COUNTS'
 import jev_oneweek_common as old
 from jev_phase13_runtime import cache_inputs
 from build_dense_jev_stage2_dataset import OfflineLabels
 oldlabels={tuple(r['key']):r for r in load_dense(read(ROOT/'reports/JEV_ONEWEEK/SUPERVISION_AUDIT.json')['labels']['path'])['labels']}
 all_labels=[];summaries=[]
 for video in TRAIN:
  source=read(old.PHASE15/'commitment_dataset_v2'/f'video{video:02d}/RESULT.json')['DATASET'];assert sha(source['path'])==source['SHA256']
  records=load_dense(source['path'])['records'];by_key={tuple(r['key'][1:]):r for r in records}
  _,_,reader=cache_inputs(video);offline=OfflineLabels(video,reader)
  for key,image in offline.images.items():
   c=collections.Counter(a['instance_id'] for a in offline.gt[image['id']]);bad={g for g,n in c.items() if n>1}
   if bad:offline.aligned[key]=[None if g in bad else g for g in offline.current(*key)]
  with gzip.open(old.PHASE15/'commitment_dataset_v2'/f'video{video:02d}/COMMITS.jsonl.gz','rt') as f:commits=[json.loads(line) for line in f]
  views=collections.defaultdict(collections.Counter);vt=collections.Counter();pooled=collections.defaultdict(collections.Counter);pt=collections.Counter();counts=collections.Counter();groups=collections.defaultdict(set)
  def update(frame,view,ids,targets):
   assert len(ids)==len(targets)
   for identity,gt in zip(ids,targets):
    vt[identity,view]+=1;pt[identity]+=1
    if gt is not None:views[identity,view][gt]+=1;pooled[identity][gt]+=1
  boot=1-int(commits[0]['key'][1]);targets=offline.current(0,boot);update(0,boot,list(range(1,len(targets)+1)),targets)
  for event in commits:
   frame,view=map(int,event['key']);r=by_key.get((frame,view));targets=offline.current(frame,view)
   if r is not None:
    assert targets==r['GT_labels_OFFLINE_ONLY'];q,k=r['inputs']['legal'][0].shape
    cross=torch.full((q,k),-1,dtype=torch.int8);continuity=cross.clone()
    for row,gt in enumerate(targets):
     if gt is None:continue
     for col,identity in enumerate(r['refs']):
      if not r['inputs']['legal'][0,row,col]:continue
      other=certify(views[identity,1-view],vt[identity,1-view])
      if other is not None:cross[row,col]=int(other==gt)
      n=sum(pooled[identity].values());owner=certify(pooled[identity],pt[identity]);own=certify(views[identity,view],vt[identity,view])
      if n>=3 and n/max(1,pt[identity])>=.8 and len(pooled[identity])>1:
       continuity[row,col]=0;counts['past_observed_mixture_certifies_continuity_violation']+=1
      elif owner is not None and owner!=gt:continuity[row,col]=0
      elif owner==gt and own is not None:continuity[row,col]=int(own==gt)
      aa,bb=int(cross[row,col]),int(continuity[row,col]);pattern='both_UNKNOWN' if aa<0 and bb<0 else 'one_UNKNOWN' if aa<0 or bb<0 else f'Cross{aa}_Continuity{bb}'
      counts[pattern]+=1
      if aa>=0 and bb>=0 and aa!=bb:
       counts['disagreement_edges']+=1
       groups['person'].add((video,gt));groups['person_time'].add((video,gt,view,frame//64));groups['ID_history'].add((video,identity));groups['conflict'].add((video,gt,identity,view,frame//64))
       if not r['known_options'][row,col]:counts['disagreements_with_association_UNKNOWN']+=1
    annotation=dict(oldlabels[tuple(r['key'])],cross=cross,stability=continuity)
    all_labels.append(annotation)
   update(frame,view,event['ids'],targets)
  summaries.append(dict(video=video,counts=dict(counts),groups={n:len(g) for n,g in groups.items()},source=source))
  save(OUT/'R2_labels/PROGRESS.json',dict(status='RUNNING',complete_videos=summaries));print('R2_LABELS_VIDEO',video,dict(counts),flush=True)
  del records,by_key,offline,reader,commits;gc.collect()
 path=OUT/'R2_labels/LABELS.pth.xz';path.parent.mkdir(exist_ok=True)
 save_dense(path,dict(labels=all_labels,protocol=ref(REPORTS/'R2_LABEL_PROTOCOL.json')),reserve=30*2**30)
 save(REPORTS/'R2_QUESTION_SUPERVISION.json',dict(status='COMPLETE',binding=binding('Current cross evidence and observed persistent ownership continuity labels'),videos=summaries,labels=ref(path),association_known_options_unchanged=True,contaminated_owner_not_relabelled=True,TRAIN16_gradient=False,future_safety=False))
 (DOCS/'SUPERVISION_REDESIGN.md').write_text('# R2 两类真实问题\n\nCross：其他摄像头过去有至少3个已知、至少80%已知覆盖且完全同人的支持，是否与当前人一致；不再要求整个 Global ID 的混合历史先认证为纯 owner。Continuity：具体 ID 的已观测归属是否支持当前人的连续性；充分观测到多个不同人认证的是“连续归属违例”，并不认证正确 owner。纯 ID 与当前人矛盾为0；纯 ID 且 own-camera 历史支持当前人为1；其余 UNKNOWN。\n\n这些是不同可验证谓词，不修改 GT、不把 owner UNKNOWN 当错误、不把历史可靠性概率称为已校准未来安全。关联 known_options 和 changed-row 比较仍使用原冻结标签；混合 ID 无法形成可靠关联排名的限制仍存在。每个视频分歧及依赖组见 R2_QUESTION_SUPERVISION.json。\n')
 print('R2_LABELS_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:failure('R2_question_supervision',traceback.format_exc());raise
