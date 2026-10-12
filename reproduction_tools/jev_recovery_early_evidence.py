"""GT-free first-three committed appearances, reconstructed without copying dense input."""
import collections,gc,gzip,torch,traceback
import torch.nn.functional as F
from jev_recovery_common import *
from jev_phase14_artifacts import load_dense,save_dense
def main():
 protect();torch.set_num_threads(1)
 protocol=read(REPORTS/'R2_EARLY_EVIDENCE_PROTOCOL.json');assert protocol['status']=='FROZEN_BEFORE_NEW_R2_GRADIENTS'
 import jev_oneweek_common as old
 from jev_phase13_runtime import cache_inputs
 patches=[];sources=[];counts=[]
 for video in TRAIN:
  folder=old.PHASE15/'commitment_dataset_v2'/f'video{video:02d}';source=read(folder/'RESULT.json')['DATASET'];assert sha(source['path'])==source['SHA256']
  records=load_dense(source['path'])['records'];lookup={tuple(r['key'][1:]):r for r in records}
  with gzip.open(folder/'COMMITS.jsonl.gz','rt') as f:commits=[json.loads(line) for line in f]
  _,_,reader=cache_inputs(video);boot=1-int(commits[0]['key'][1]);payload=reader.load(video,0,boot)
  first=collections.defaultdict(list)
  for row,feat in enumerate(F.normalize(payload['reid_features'][:,:1024],dim=-1)):first[row+1].append(feat)
  total=0
  for event in commits:
   frame,view=map(int,event['key']);r=lookup.get((frame,view))
   if r is not None:
    k=len(r['refs']);proto=torch.zeros(k,1024);mask=torch.zeros(k,dtype=torch.bool)
    for col,identity in enumerate(r['refs']):
     if first[identity]:proto[col]=F.normalize(torch.stack(first[identity]).mean(0),dim=-1);mask[col]=True
    patches.append(dict(key=r['key'],anchor_visual=proto,anchor_mask=mask));total+=int(mask.sum())
    features=r['inputs']['detection_visual'][0]
   else:features=F.normalize(reader.load(video,frame,view)['reid_features'][:,:1024],dim=-1)
   assert len(features)==len(event['ids'])
   for identity,feature in zip(event['ids'],features):
    if len(first[identity])<3:first[identity].append(feature.clone())
  counts.append(dict(video=video,payloads=len(records),past_native_ID_prototypes=total));sources.append(source)
  save(OUT/'early_evidence/PROGRESS.json',dict(status='RUNNING',complete_videos=counts));print('EARLY_NATIVE_EVIDENCE',video,total,flush=True)
  del records,lookup,reader,commits;gc.collect()
 p=OUT/'early_evidence/PATCH.pth.xz';p.parent.mkdir(exist_ok=True);save_dense(p,dict(patches=patches,protocol=ref(REPORTS/'R2_EARLY_EVIDENCE_PROTOCOL.json')),reserve=30*2**30)
 save(REPORTS/'R2_EARLY_EVIDENCE.json',dict(status='COMPLETE',binding=binding('GT-free earliest three actual committed normalized appearances'),patch=ref(p),sources=sources,counts=counts,GT_used=False,actor_candidates_unchanged=True,memory_admission_unchanged=True,extra_perception=False,attribution='Added persistence evidence, offered to all ordinary controls; not intrinsically a JEV architecture advantage'))
 print('EARLY_EVIDENCE_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:failure('early_native_evidence',traceback.format_exc());raise
