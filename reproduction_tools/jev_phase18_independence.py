"""Prospective independence qualification; no resampling favorable outcomes."""
from jev_phase18_common import *
if __name__=='__main__':
 protect();events=read(OUT/'selection/INTERVENTION_EVENTS.json');parent=list(range(len(events)))
 def find(i):
  if parent[i]!=i:parent[i]=find(parent[i])
  return parent[i]
 def union(a,b):
  a=find(a);b=find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 for i,a in enumerate(events):
  for j,b in enumerate(events[:i]):
   if a['group']['video']==b['group']['video'] and set(a['group']['GT_component_OFFLINE'])&set(b['group']['GT_component_OFFLINE']):union(i,j)
 components={}
 for i,e in enumerate(events):components.setdefault(str(find(i)),[]).append(i)
 leaks=[dict(component=k,members=idx,TRAIN=[i for i in idx if events[i]['split']=='TRAIN'],RESERVED=[i for i in idx if events[i]['split']=='RESERVED']) for k,idx in components.items() if any(events[i]['split']=='TRAIN' for i in idx) and any(events[i]['split']=='RESERVED' for i in idx)]
 save(REPORTS/'INDEPENDENCE_QUALIFICATION_PROTOCOL.json',dict(status='FROZEN_BEFORE_MAIN_COUNTERFACTUAL_OUTCOMES',purpose='64-frame blocks are sampling units, not automatically independent episodes/owners. Same-video overlapping immutable owner/alias/conflict GT components across time remain one conservative qualification unit.',rule='Merge selected components across all time on physical owner/conflict overlap; any mixed TRAIN/RESERVED component is contaminated and cannot supply a reserved safe group. No resampling after outcomes.',intervention_events=ref(OUT/'selection/INTERVENTION_EVENTS.json'),qualification_components=components,pre_outcome_reserved_leakage=leaks,original16_rollouts_still_execute_for_causal_diagnostics=True,pre_outcome_reserved_qualification='FAIL_INDEPENDENCE' if leaks else 'PENDING_SAFE_FUTURES',GT_only_offline_partition_audit=True,no_new_training=True))
 print('PROSPECTIVE_COMPONENTS',len(components),'mixed train/reserved',len(leaks))
