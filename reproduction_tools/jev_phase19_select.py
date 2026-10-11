"""Prospective fixed-video sampling; full current-trajectory census, no reward selection."""
from jev_phase19_common import *
import gzip,collections,random
def rows(path):
 with gzip.open(path,'rt') as f:return [json.loads(l) for l in f]
def main():
 protect();preflight=read(REPORTS/'SCOPE_GT_CACHE_PREFLIGHT.json');meta={v['video']:v for v in preflight['per_video']};parent={};all_events=[];sources=[];counts={}
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def union(a,b):
  a=find(a);b=find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 for v in SCOPE:
  path=census_case(v)/'RESULT.json';r=read(path);assert r['status']=='COMPLETE';sources.append(ref(path));events=rows(r['events']['path']);counts[str(v)]=r['counts'];all_events+=events
  for e in events:
   gt=e['GT_OFFLINE']
   if gt is None:continue
   block=e['event_key'][1]//64;a=(v,block,gt);find(a);conflicts=list(e['numeric_close_owner_conflict_OFFLINE'])
   if e['wrong_owner'] and e['owner_before_commit_OFFLINE'] is not None:conflicts.append(e['owner_before_commit_OFFLINE'])
   for other in conflicts:union(a,(v,block,other))
 members=collections.defaultdict(set)
 for node in parent:members[find(node)].add(node[2])
 groups=collections.defaultdict(list);eligibility=collections.Counter()
 for e in all_events:
  v,f,cam,row=e['event_key']
  if e['GT_OFFLINE'] is None:continue
  complete=f+64<=meta[v]['scene_frames'];unique=all(d['frame_zero_based']>=f+64 for d in meta[v]['actual_duplicates']);has_alts=bool(e['alternatives_GT_free'])
  eligible=complete and unique and has_alts;eligibility[str(v)+'_eligible']+=eligible;eligibility[str(v)+'_ineligible']+=not eligible
  if eligible:groups[find((v,f//64,e['GT_OFFLINE']))].append(e)
 reps={};pop=collections.defaultdict(list)
 for g,events in groups.items():
  for label in ['AVAILABLE_ERROR','CORRECT_CONTROL']:
   candidates=sorted([e for e in events if e['category']==label],key=lambda e:e['event_key'])
   if candidates:reps[g]=candidates[0];pop[g[0],label].append(g);break
 sampled=[];drawn={};strata=[]
 for v in SCOPE:
  for label in ['AVAILABLE_ERROR','CORRECT_CONTROL']:
   population=sorted(pop[v,label]);draw=population.copy();random.Random(20261009+100*v+(label=='CORRECT_CONTROL')).shuffle(draw);quota=2 if v==15 else 3;draw=draw[:quota];drawn[v,label]=draw
   for g in draw:
    e=dict(reps[g]);e.update(group=dict(video=v,time_block=g[1],GT_component_OFFLINE=sorted(members[g])),sampling_probability=min(1,quota/max(1,len(population))),sampled_stratum=label,this_round_split='TRAIN' if v in TRAIN else 'RESERVED',strict_H64_eligible_before_outcomes=True);sampled.append(e)
   strata.append(dict(video=v,split='TRAIN' if v in TRAIN else 'RESERVED',stratum=label,population=len(population),sampled=len(draw),quota=quota))
 def sample_event(g):
  return next(e for e in sampled if e['group']==dict(video=g[0],time_block=g[1],GT_component_OFFLINE=sorted(members[g])))
 selected=[];selected_g=set()
 for v in SCOPE:
  keys=drawn[v,'AVAILABLE_ERROR'].copy();random.Random(20261019+v).shuffle(keys)
  if keys:selected.append(sample_event(keys[0]));selected_g.add(keys[0])
 remaining=[g for v in TRAIN for g in drawn[v,'AVAILABLE_ERROR'] if g not in selected_g];random.Random(20261019).shuffle(remaining)
 while sum(e['this_round_split']=='TRAIN' for e in selected)<9 and remaining and len(selected)<12:
  g=remaining.pop();selected.append(sample_event(g));selected_g.add(g)
 for split,quota in [('TRAIN',3),('RESERVED',1)]:
  pool=[g for v in SCOPE if (v in TRAIN)==(split=='TRAIN') for g in drawn[v,'CORRECT_CONTROL']];random.Random(20261029+(split=='RESERVED')).shuffle(pool)
  selected.extend(sample_event(g) for g in pool[:quota])
 intervention=[dict(e,selection_index=i,split=e['this_round_split'],pre_reserved=e['this_round_split']=='RESERVED') for i,e in enumerate(selected)]
 assert len(sampled)<=64 and len(intervention)<=16
 save(OUT/'selection/SAMPLED_EVENTS.json',sampled);save(OUT/'selection/INTERVENTION_EVENTS.json',intervention)
 # Cross-time overlap is qualified before action outcomes; split is video-fixed.
 parents=list(range(len(intervention)))
 def ff(i):
  if parents[i]!=i:parents[i]=ff(parents[i])
  return parents[i]
 for i,e in enumerate(intervention):
  for j,b in enumerate(intervention[:i]):
   if e['group']['video']==b['group']['video'] and set(e['group']['GT_component_OFFLINE'])&set(b['group']['GT_component_OFFLINE']):parents[ff(i)]=ff(j)
 components={}
 for i,e in enumerate(intervention):components.setdefault(str(ff(i)),[]).append(i)
 assert all(len({intervention[i]['split'] for i in ids})==1 for ids in components.values())
 save(REPORTS/'INDEPENDENCE_QUALIFICATION_PROTOCOL.json',dict(status='FROZEN_BEFORE_ACTION_OUTCOMES',qualification_components=components,pre_outcome_reserved_leakage=[],intervention_events=ref(OUT/'selection/INTERVENTION_EVENTS.json'),same_video_cross_time_owner_conflicts_merged=True,reserved_is_historically_exposed_validation=True,no_fresh_test_claim=True,no_resampling_after_outcomes=True))
 save(REPORTS/'COMPLETE_NATIVE_OWNER_CENSUS.json',dict(status='COMPLETE',binding=binding(),sources=sources,per_video_counts=counts,eligible_event_counts=dict(eligibility),strata=strata,sampled=ref(OUT/'selection/SAMPLED_EVENTS.json'),intervention_events=ref(OUT/'selection/INTERVENTION_EVENTS.json'),intervention_groups=len(intervention),complete_native_branch_budget=sum(1+len(e['alternatives_GT_free']) for e in intervention),qualified_safe_groups=None,representation='earliest eligible class event per timeblock conflict group; video-stratified uniform quota, no reward-based resampling',strict_GT_filter='complete64frame future and rawGT unique from frame0 to futureend; no raw rows rewritten',historical_exposure=ref(REPORTS/'HISTORICAL_EXPOSURE_AUDIT.json')))
 print('PHASE19_SELECTION_COMPLETE',len(sampled),len(intervention),components,flush=True)
if __name__=='__main__':main()

