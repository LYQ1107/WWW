"""Lossless factual-prefix translation from replay writes to native galleries.

Replay memory contains post-birth writes; native gallery also has initial birth
observation. Reconstruct that real cached vector, never fabricate an embedding.
This translation changes neither historical rollouts nor their action labels.
"""
import gzip,json
import torch
from jev_phase9_common import PREVIOUS,sha
class NativeGalleryBridge:
 def __init__(self,lab):
  self.lab=lab;self.first={}
  seed=lab.seed_state.association_history[0]
  for r,t in seed['assignments'].items():self.first[int(t)]={'key':tuple(lab.seed_key),'row':int(r),'seed':True}
  path=PREVIOUS/'opportunity_scan_v1'/f'video{lab.video:02d}'/'natural_event_index.jsonl.gz'
  for d in map(json.loads,gzip.open(path,'rt')):
   self.first.setdefault(int(d['actual_committed_id']),{'key':tuple(d['key']),'row':int(d['row']),'seed':False})
  self.source={'path':str(path),'sha256':sha(path)}
 def galleries(self,state,key):
  result={}
  for track,hits in state.track_hits.items():
   track=int(track);birth=self.first[track];assert birth['key'][1]<=key[1]
   assert len(state.memory.get(track,[]))==hits-1,('not the factual all-WRITE prefix representation',track,hits,len(state.memory.get(track,[])))
   payload=self.lab.cache.load(*birth['key']);first=payload['reid_features'][birth['row']].detach().cpu()
   result[track]=torch.stack([first]+list(state.memory.get(track,[])))
  return result
