"""Frozen conflict components, uniform group sampling and prior reservation."""
from jev_phase18_common import *
import gzip,collections,random

def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def main():
 protect();all_events=[];sources=[];totals={};parent={}
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def union(a,b):
  a=find(a);b=find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 for v in TRAIN+DIAGNOSTIC:
  path=OUT/'census_v2'/f'video{v:02d}/RESULT.json';r=read(path);assert r['status']=='COMPLETE';assert r['binding']['source_commit']=='006211cb32a03b5c4e24234e660d359e473d4dbd';sources.append(ref(path));totals[str(v)]=r['counts'];events=rows(r['events']['path']);all_events+=events
  if v not in TRAIN:continue
  for e in events:
   gt=e['GT_OFFLINE']
   if gt is None:continue
   block=e['event_key'][1]//64;a=(v,block,gt);find(a);conflicts=list(e['numeric_close_owner_conflict_OFFLINE'])
   if e['wrong_owner'] and e['owner_before_commit_OFFLINE'] is not None:conflicts.append(e['owner_before_commit_OFFLINE'])
   for other in conflicts:union(a,(v,block,other))
 members=collections.defaultdict(set)
 for node in parent:members[find(node)].add(node[2])
 groups=collections.defaultdict(list)
 for e in all_events:
  if not e['strict'] or e['GT_OFFLINE'] is None:continue
  v,frame,view,row=e['event_key'];groups[find((v,frame//64,e['GT_OFFLINE']))].append(e)
 strata=collections.defaultdict(list);priority=['AVAILABLE_ERROR','UNAVAILABLE_ERROR','CORRECT_CONTROL'];representatives={}
 for g,events in groups.items():
  for label in priority:
   items=sorted([e for e in events if e['category']==label],key=lambda e:e['event_key'])
   if items:strata[label].append(g);representatives[g]=items[0];break
 quotas=dict(AVAILABLE_ERROR=24,UNAVAILABLE_ERROR=8,CORRECT_CONTROL=32);sampled=[];chosen={};report=[]
 for i,label in enumerate(priority):
  population=sorted(strata[label]);rng=random.Random(20261009+i);draw=population.copy();rng.shuffle(draw);draw=draw[:quotas[label]];chosen[label]=draw
  for g in draw:
   event=dict(representatives[g]);event.update(group=dict(video=g[0],time_block=g[1],GT_component_OFFLINE=sorted(members[g])),sampling_probability=min(1,quotas[label]/max(1,len(population))),sampled_stratum=label);sampled.append(event)
  report.append(dict(stratum=label,population_groups=len(population),sampled_groups=len(draw),quota=quotas[label]))
 intervention=[]
 for i,(label,n,reserved) in enumerate([('AVAILABLE_ERROR',12,3),('CORRECT_CONTROL',4,1)]):
  keys=chosen[label].copy();random.Random(20261019+i).shuffle(keys);keys=keys[:n]
  for j,g in enumerate(keys):
   e=next(e for e in sampled if e['group']['video']==g[0] and e['group']['time_block']==g[1] and e['group']['GT_component_OFFLINE']==sorted(members[g]));item=dict(e);item.update(pre_reserved=j<reserved,split='RESERVED' if j<reserved else 'TRAIN',selection_index=len(intervention));intervention.append(item)
 save(OUT/'selection/SAMPLED_EVENTS.json',sampled);save(OUT/'selection/INTERVENTION_EVENTS.json',intervention)
 save(REPORTS/'COMPLETE_NATIVE_OWNER_CENSUS.json',dict(status='COMPLETE',binding=binding(),sources=sources,per_video_counts=totals,grouping='cross-camera currentGT, immutable wrong owner and numeric close competitors union within64-frame block; conflict components separate from individual candidate pairs',strata=report,priority='A component occurs once: available_error > unavailable_error > correct_control; earliest event of assigned class',total_components=len(groups),sampled=ref(OUT/'selection/SAMPLED_EVENTS.json'),intervention_events=ref(OUT/'selection/INTERVENTION_EVENTS.json'),intervention_groups=len(intervention),pre_reserved_error_groups=sum(e['pre_reserved'] and e['sampled_stratum']=='AVAILABLE_ERROR' for e in intervention),pre_reserved_control_groups=sum(e['pre_reserved'] and e['sampled_stratum']=='CORRECT_CONTROL' for e in intervention),numeric_candidate_recall_scope='per event full lawful pool, unknown correct owner availability is null; no Top16 truncation',no_teacher_or_learning=True,eligible_safe_groups=None))
 print('CENSUS_SELECTION_COMPLETE',totals,report,len(intervention),flush=True)
if __name__=='__main__':main()
