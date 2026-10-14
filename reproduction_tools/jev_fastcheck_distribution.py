"""No numeric-ID mismatch shortcut: compare past-certified owners and coverage."""
import collections,gzip,lzma,torch
from jev_fastcheck_common import *
def records(result):
 for cp in result['record_chunks']:
  assert sha(cp['path'])==cp['SHA256']
  with lzma.open(cp['path'],'rb') as f:
   for r in torch.load(f,map_location='cpu')['records']:yield r
def summarize(result):
 c=collections.Counter();groups=set();people=set();episodes=[];timelines=collections.defaultdict(dict)
 for r in records(result):
  c['payloads']+=1;k=len(r['refs']);legal=r['inputs']['legal'][0]
  for row,g in enumerate(r['GT_labels_OFFLINE_ONLY']):
   c['rows']+=1;c[r['eligibility_categories'][row]]+=1
   c['unbound_legal_options']+=int((legal[row]&~r['known_options'][row,:k]).sum())
   c['certified_active_support']+=int(bool(r['positive'][row,:k].any()))
   if g is not None:groups.add((r['key'][0],g,r['key'][2],r['key'][1]//64));people.add((r['key'][0],g))
   if r['positive'][row,:k].any():
    selected=r['native_choice'][row];col=k if selected<0 else selected
    c['correct_identity_protection']+=int(bool(r['positive'][row,col]))
    c['supported_but_wrong_known']+=int(bool(r['known_options'][row,col]) and not bool(r['positive'][row,col]))
    c['supported_but_UNKNOWN']+=int(not bool(r['known_options'][row,col]))
   for p in r['plans']:
    chosen=[k if x<0 else x for x in p]
    if all(r['known_options'][row,col] for row,col in enumerate(chosen)):c['fully_certified_plans']+=1
 for line in gzip.open(result['journal']['path'],'rt'):
  event=json.loads(line);frame,view=event['key'][1:]
  for row,(g,tag) in enumerate(zip(event['GT_OFFLINE_ONLY'],event['owner_tags'])):
   if g is not None:timelines[g,view][frame]=tag=='WRONG'
 for key,timeline in timelines.items():
  start=None;prev=None
  for f,bad in sorted(timeline.items()):
   if start is not None and (not bad or f!=prev+1):episodes.append(dict(person=key[0],camera=key[1],start=start,end=prev,duration=prev-start+1));start=None
   if bad and start is None:start=f
   prev=f
  if start is not None:episodes.append(dict(person=key[0],camera=key[1],start=start,end=prev,duration=prev-start+1,right_censored=True))
 return dict(counts=dict(c),local_person_camera_time_groups=len(groups),dependent_person_components=len(people),mixed_IDs=result['mixed_IDs'],known_wrong_duration_camera_frames=sum(e['duration'] for e in episodes),episodes=episodes,
  coverage=c['certified_active_support']/max(1,c['rows']-c['UNKNOWN_CURRENT']),certificate_scope='first past-certified ID; not birth-owner or future-safe control')
def main():
 rows=[];first=[];eligibility=[]
 for video in [12,13,14]:
  teacher=read(REPORTS/f'ROLLOUT_teacher_VIDEO{video:02d}.json');self=read(REPORTS/f'ROLLOUT_self_VIDEO{video:02d}.json')
  a=read(OUT/'rollouts/teacher'/f'video{video:02d}/START.json');b=read(OUT/'rollouts/self'/f'video{video:02d}/START.json')
  assert a['cache_index']==b['cache_index'];assert teacher['initial_input_SHA']==self['initial_input_SHA'];assert teacher['initial_native_state_SHA']==self['initial_native_state_SHA']
  tj={tuple(e['key']):e for e in map(json.loads,gzip.open(teacher['journal']['path'],'rt'))}
  sj={tuple(e['key']):e for e in map(json.loads,gzip.open(self['journal']['path'],'rt'))}
  divergence=next((key for key in sorted(tj) if tj[key]['ids']!=sj[key]['ids']),None)
  # Before the first ID/state divergence, IDs share the exact common prefix.
  # Later row comparisons use offline owner tags and actual person annotations.
  first.append(dict(video=video,key=divergence,teacher=tj.get(divergence),self=sj.get(divergence),
   first_difference_identical_prefix=True,subsequent_ID_numbers_not_compared_for_correctness=True))
  aa=summarize(teacher);bb=summarize(self);rows.append(dict(video=video,teacher=aa,self=bb,same_exogenous_cache=True,same_initial_state=True))
  eligibility.append(dict(video=video,status='CURRENT_MATCH_LABELS_ONLY',counts=bb['counts'],groups=bb['local_person_camera_time_groups'],dependent_components=bb['dependent_person_components'],no_long_term_safety_qualification=True))
 result=dict(status='COMPLETE_BOUNDED_P1',rows=rows,first_divergences=first,H1='actual state distribution differences measured; not established as unique causal cause of degradation',independent_validation=False,source=binding('P1 actual self/teacher state comparison'))
 save(REPORTS/'STATE_DISTRIBUTION_AUDIT.json',result)
 save(REPORTS/'SELF_POLICY_ROLLOUT.json',dict(status='COMPLETE_FROZEN_SELF_NATIVE',sources=[ref(REPORTS/f'ROLLOUT_self_VIDEO{v:02d}.json') for v in [12,13,14]],no_teacher_state_copy=True))
 save(REPORTS/'TRAIN_SAMPLE_ELIGIBILITY.json',dict(status='ELIGIBLE_CURRENT_MATCH_ONLY',rows=eligibility,UNKNOWN_not_forced=True,dependent_events_not_IID=True,first_past_certificate_not_future_safety=True))
 lines=['# Native state distribution audit','','Same Stage1 indexes, identical first inputs and complete initial native-state fingerprint matched for all pairs. All future Gallery/Bank/candidate states were recomputed independently.','','| Video | Teacher mixed IDs | Self mixed IDs | Teacher wrong duration | Self wrong duration | Teacher coverage | Self coverage |','|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  a,b=r['teacher'],r['self'];lines.append(f"| {r['video']} | {a['mixed_IDs']} | {b['mixed_IDs']} | {a['known_wrong_duration_camera_frames']} | {b['known_wrong_duration_camera_frames']} | {a['coverage']:.4f} | {b['coverage']:.4f} |")
 lines+=['','These first256-frame TRAIN prefixes demonstrate differences in visited states, not a causal proof that mismatch explains full-video degeneration. Local groups are dependent by person/history; positive labels mean first-past certified current MATCH preservation, not future safety. P2 tests the data intervention.']
 (DOCS/'STATE_DISTRIBUTION_REPORT.md').write_text('\n'.join(lines)+'\n')
 (DOCS/'FIRST_DIVERGENCE_REPORT.md').write_text('# First native divergence\n\n'+ '\n\n'.join(f"Video {f['video']}: first changed native commit at `{f['key']}`. Both trajectories share the same entire starting state and identical decisions up to this commit. Prior owner tags: teacher `{f['teacher']['owner_tags'] if f['teacher'] else None}`, self `{f['self']['owner_tags'] if f['self'] else None}`. Post-divergence IDs require person/certificate alignment, not direct numeric equality." for f in first)+'\n\nFull details, actual events, Gallery lengths and Bank IDs are retained in STATE_DISTRIBUTION_AUDIT.json and immutable journals.\n')
 print('P1_COMPLETE',[(r['video'],r['teacher']['mixed_IDs'],r['self']['mixed_IDs']) for r in rows],flush=True)
if __name__=='__main__':main()
