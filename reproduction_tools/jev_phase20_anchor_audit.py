"""Score fixed events before their actual camera commits using first3 raw vectors."""
from jev_phase20_common import *
import gzip,collections
import numpy as np
import torch
import torch.nn.functional as F


def lines(p):
 with gzip.open(p,'rt') as f:
  for line in f:yield json.loads(line)


def score_candidates(query,refs,legal,original_cosine,ledger):
 """Actor-side calculation has no GT, outcome labels or future observations."""
 query=F.normalize(query[:1024],dim=-1)
 scores=[];support=[]
 for refid,old in zip(refs,original_cosine):
  past=ledger.get(refid,[]);n=len(past);support.append(n)
  if n<3:scores.append(old);continue
  proto=torch.stack([x['vector'] for x in past]).mean(0)
  scores.append(float(query@F.normalize(proto,dim=-1)) if float(proto.norm())>0 else old)
 ranks=sorted([i for i,ok in enumerate(legal) if ok],key=lambda i:(-scores[i],refs[i]))
 return scores,support,ranks


def dependency_components(events):
 """Offline grouping only; same full-census conflict rule as the prior phase."""
 parent={}
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def union(a,b):
  a=find(a);b=find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 for v in SCOPE:
  source=census(v)['events'];assert sha(source['path'])==source['SHA256']
  for e in lines(source['path']):
   gt=e['GT_OFFLINE']
   if gt is None:continue
   a=(v,gt);find(a);conflicts=list(e['numeric_close_owner_conflict_OFFLINE'])
   if e['wrong_owner'] and e['owner_before_commit_OFFLINE'] is not None:conflicts.append(e['owner_before_commit_OFFLINE'])
   for other in conflicts:union(a,(v,other))
 return {i:str(find((e['event_key'][0],e['GT_OFFLINE']))) for i,e in enumerate(events)}


def main():
 protect();torch.set_num_threads(1);started=time.monotonic()
 assert not (OUT/'RESULT.json').exists(),'immutable run already exists'
 protocol=read(REPORTS/'PREREGISTRATION.json');population=protocol['population']
 assert sha(population['path'])==population['SHA256'];events=read(population['path']);assert len(events)==50
 components=dependency_components(events)
 all_results=[];sources=[];vector_sources={};payload_hashes={};prototype_log=[];loaded=0
 for video in SCOPE:
  selected=[(i,e) for i,e in enumerate(events) if e['event_key'][0]==video]
  if not selected:continue
  c=census(video);sources.extend([c['commits'],c['events']])
  for s in [c['commits'],c['events']]:assert sha(s['path'])==s['SHA256']
  cache=PRIOR_OUT/'stage1_cache_v1'/f'video{video:02d}'
  index={(r['frame'],r['view']):r for r in map(json.loads,(cache/'index.jsonl').read_text().splitlines())}
  commits={tuple(r['key']):r['ids'] for r in lines(c['commits']['path'])}
  counts=[index[0,v]['detection_count'] for v in [0,1]];bootstrap=int(np.argsort(counts)[::-1][0])
  assert {(f,v) for f in range(c['frames']) for v in [0,1]}-set(commits)=={(0,bootstrap)}
  commits[0,bootstrap]=list(range(1,counts[bootstrap]+1))
  if c.get('recovery'):assert c['recovery']['bootstrap']['key']==[0,bootstrap] and c['recovery']['bootstrap']['IDs']==commits[0,bootstrap]
  wanted=collections.defaultdict(list)
  for i,e in selected:wanted[tuple(e['event_key'][1:3])].append((i,e))
  ledger={};stop=max(k[0] for k in wanted);ordering=[(0,bootstrap),(0,1-bootstrap)]+[(f,v) for f in range(1,stop+1) for v in [0,1]]
  # Each payload is loaded only for selected queries or unfinished first3 histories.
  for key in ordering:
   ids=commits[key];needed=key in wanted or any(len(ledger.get(t,[]))<3 for t in ids)
   if not needed:continue
   record=index[key];p=cache/record['record'];digest=record['record_sha256'].removeprefix('sha256:')
   assert sha(p)==digest;payload_hashes[str(p)]=digest
   payload=torch.load(p,map_location='cpu');features=payload['reid_features']
   assert len(features)==len(ids);loaded+=1
   # Every row in this camera shares the same prior state, before any row commits.
   for i,e in wanted.get(key,[]):
    row=e['event_key'][3]
    assert ids[row]==e['actual_committed_ref']
    for r in e['refs']:
     assert ledger.get(r),'candidate not represented in actual prior commits'
     for past in ledger[r]:
      pf,pc=past['event_key'][1:3]
      assert pf<key[0] or pf==key[0] and (pf==0 and pc==bootstrap and pc!=key[1] or pf>0 and pc<key[1]),'current/future feature leaked into candidate'
    scores,support,ranks=score_candidates(features[row],e['refs'],e['legal'],e['cosine'],ledger)
    oldr=sorted([j for j,ok in enumerate(e['legal']) if ok],key=lambda j:(-e['cosine'][j],e['refs'][j]))
    old=oldr[0] if oldr else None;new=ranks[0] if ranks else None
    # Offline evaluator reads the immutable historical owner labels only now.
    gt=e['GT_OFFLINE'];owners=e['owner_GT_OFFLINE']
    def label(j):return 'UNKNOWN' if j is None or gt is None or owners[j] is None else ('CORRECT' if owners[j]==gt else 'WRONG')
    correct=set(e['correct_owner_refs_OFFLINE'])
    row_result=dict(sample_index=i,event_key=e['event_key'],split=e['this_round_split'],stratum=e['sampled_stratum'],
     dependency_component_OFFLINE=components[i],actual_committed_ref=e['actual_committed_ref'],refs=e['refs'],legal=e['legal'],scores=scores,
     baseline_scores=e['cosine'],support_first3=support,
     baseline_top_ref=None if old is None else e['refs'][old],anchor_top_ref=None if new is None else e['refs'][new],
     baseline_owner_label_OFFLINE=label(old),anchor_owner_label_OFFLINE=label(new),
     correct_owner_ranks_OFFLINE=dict(baseline=next((n+1 for n,j in enumerate(oldr) if e['refs'][j] in correct),None),
       anchor=next((n+1 for n,j in enumerate(ranks) if e['refs'][j] in correct),None)),
     changed_choice=old!=new,legal_anchor_support=sum(ok and n==3 for ok,n in zip(e['legal'],support)),
     legal_candidates=sum(e['legal']),mixed_history=e['mixed_history'],
     anchor_top_mixed_history=None if new is None else e['mixed_history'][new],
     selected_score_margin=None if len(ranks)<2 else scores[ranks[0]]-scores[ranks[1]],
     GT_actor_input=False,pointwise_only=True,assignment_or_commit_modified=False)
    all_results.append(row_result)
   for row,t in enumerate(ids):
    past=ledger.setdefault(t,[])
    if len(past)<3:
     origin=dict(event_key=[video,*key,row],payload=ref(p),identity=t,bootstrap=key==(0,bootstrap))
     past.append(dict(vector=features[row,:1024].clone(),**origin))
     vector_sources[video,t]=len(past);prototype_log.append(origin)
  save(OUT/'PROGRESS.json',dict(status='RUNNING',video=video,completed_events=len(all_results),loaded_payloads=loaded,seconds=time.monotonic()-started))
  print('ANCHOR_AUDIT_VIDEO_DONE',video,len(selected),flush=True)
 assert len(all_results)==50
 all_results.sort(key=lambda r:r['sample_index'])
 def summarize(items):
  transitions=collections.Counter((e['baseline_owner_label_OFFLINE']+'_to_'+e['anchor_owner_label_OFFLINE']) for e in items)
  return dict(events=len(items),baseline_correct=sum(e['baseline_owner_label_OFFLINE']=='CORRECT' for e in items),
   anchor_correct=sum(e['anchor_owner_label_OFFLINE']=='CORRECT' for e in items),
   baseline_known=sum(e['baseline_owner_label_OFFLINE']!='UNKNOWN' for e in items),
   anchor_known=sum(e['anchor_owner_label_OFFLINE']!='UNKNOWN' for e in items),
   choice_changes=sum(e['changed_choice'] for e in items),transitions=dict(transitions),
   full_legal_anchor_coverage_events=sum(e['legal_anchor_support']==e['legal_candidates'] for e in items),
   dependency_units=len({e['dependency_component_OFFLINE'] for e in items}))
 strata={f'{split}_{label}':summarize([e for e in all_results if e['split']==split and e['stratum']==label])
  for split in ['TRAIN','RESERVED'] for label in ['AVAILABLE_ERROR','CORRECT_CONTROL']}
 save(OUT/'PROTOTYPE_PROVENANCE.json',dict(raw_actual_commit_vectors=prototype_log,payload_hashes=payload_hashes,
  no_GT_selection=True,stored_vectors_in_Git=False,ordering='query before allcurrentcamera rows; f0largestcamera bootstrap then other; f>0camera0then1'))
 result=dict(status='COMPLETE_EXPLORATORY_DIAGNOSTIC',binding=binding(),population=population,
  prototype_provenance=ref(OUT/'PROTOTYPE_PROVENANCE.json'),sources=sources,summary=summarize(all_results),
  strata=strata,per_video={str(v):summarize([e for e in all_results if e['event_key'][0]==v]) for v in SCOPE},
  events=all_results,loaded_payloads=loaded,seconds=time.monotonic()-started,
  dependency_units={component:summarize([e for e in all_results if e['dependency_component_OFFLINE']==component]) for component in sorted(set(components.values()))},
  new_training='NOT_RUN',new_native_actions=0,new_policy_metrics=None,statistical_significance=None,
  limitations=['Original prior native outcomes known; exploratory representation check, not independent confirmation.',
   'Pointwise ranks have no joint assignment capacity constraint; not native controller gain or safety.',
   'Earliest observations do not certify identity ownership. Frozen owner labels remain estimates.',
   'First3 may contain sameframe crosscamera observations; evidence causal but not independent.',
   'Insufficient first3 support falls back to original4slot score; mixed score is explicitly disclosed.',
   'Correlated sample groups, no resampling, no architecture comparison or training eligibility from this diagnostic.'])
 save(OUT/'RESULT.json',result);save(REPORTS/'PAST_ANCHOR_EVIDENCE_AUDIT.json',result)
 save(OUT/'PROGRESS.json',dict(status='COMPLETE',events=50,seconds=result['seconds']))
 protect();print(json.dumps(dict(summary=result['summary'],strata=strata),indent=2))


if __name__=='__main__':main()
