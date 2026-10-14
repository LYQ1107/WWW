"""Replay saved native commits offline for error duration; no new tracker run."""
import argparse,collections,gzip,torch
from jev_fastcheck_common import *
def main(references=False):
 from jev_fastcheck_rollout import LabelState
 from jev_phase13_runtime import cache_inputs
 torch.set_num_threads(1);_,frames,reader=cache_inputs(16);shared=LabelState(16,reader).labels
 for f in range(frames):
  for v in [0,1]:shared.current(f,v)
 result=[]
 paths=list(sorted(REPORTS.glob('ONLINE_direct*_STEPS1000.json')))
 if references:paths+=list(sorted((ROOT/'reports/JEV_RECOVERY/R3_XIII').glob('ONLINE_*_pilot_SEED20261009_VIDEO16.json')))
 for p in paths:
  original=read(p);tag=original.get('tag','reference_'+original.get('arm',''))
  journal=OUT/'online'/tag/'steps1000/cached/COMMITS.jsonl.gz' if 'tag' in original else pathlib.Path(original['commits']['path'])
  state=LabelState(16,reader);state.labels=shared;events=list(map(json.loads,gzip.open(journal,'rt')));state.boot(0,1-events[0]['key'][2])
  timeline=collections.defaultdict(dict);wrong_views=collections.defaultdict(set);counts=collections.Counter()
  for e in events:
   frame,view=e['key'][1:];gt=shared.current(frame,view)
   for identity,g in zip(e['ids'],gt):
    known=g is not None and identity in state.owner;bad=known and state.owner[identity]!=g
    counts['certified_known_observations']+=known;counts['certified_wrong_observations']+=bad
    if g is not None:timeline[g,view][frame]=bad or timeline[g,view].get(frame,False)
    if bad:wrong_views[identity].add(view)
   state.update(frame,view,e['ids'])
  if 'certified_first_past_owner_wrong_observations' in original:
   assert counts['certified_wrong_observations']==original['certified_first_past_owner_wrong_observations'],(tag,counts,original['certified_first_past_owner_wrong_observations'])
  episodes=[]
  for (g,v),t in sorted(timeline.items()):
   active=None;prior=None
   for f,bad in sorted(t.items()):
    if active is not None and (not bad or f!=prior+1):episodes.append(dict(person=g,camera=v,start=active,end=prior,duration_camera_frames=prior-active+1,right_censored=f!=prior+1));active=None
    if bad and active is None:active=f
    prior=f
   if active is not None:episodes.append(dict(person=g,camera=v,start=active,end=prior,duration_camera_frames=prior-active+1,right_censored=True))
  result.append(dict(tag=tag,counts=dict(counts),certified_error_duration_camera_frames=sum(e['duration_camera_frames'] for e in episodes),episodes=episodes,
   certified_wrong_IDs_reaching_both_cameras=sum(len(v)==2 for v in wrong_views.values()),journal=ref(journal),source=ref(p),
   definition='First actual past pure >=3 known and >=80% known coverage ID/person certificate; errors compare current offline GT against that fixed certificate. Durations group consecutive observed frames per person/camera; absent/UNKNOWN observations censor, not fabricated correctness. Not true birth-owner or a future-safety certificate.'))
 save(REPORTS/'CERTIFIED_ERROR_PROPAGATION_DIRECT_AND_REFERENCES.json',dict(status='COMPLETE_EXACT_JOURNAL_REPLAY',cases=result,no_new_rollout=True,GT_actor=False,raw_GT_unchanged=True))
 print('CERTIFIED_DURATION_COMPLETE',[(r['tag'],r['certified_error_duration_camera_frames'],r['certified_wrong_IDs_reaching_both_cameras']) for r in result],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--references',action='store_true');main(p.parse_args().references)
