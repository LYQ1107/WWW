"""Paired native outcome attribution; overlapping futures are not independent trials."""
from jev_phase18_common import *
import gzip,collections

def rows(path):
 with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]
def indexed(trace,boundary,h):
 return {tuple(r['key'])+(r['row'],):r for r in trace if boundary<=tuple(r['key']) and r['key'][0]<boundary[0]+h}
def episode_max(consequence):
 result=collections.defaultdict(int)
 for e in consequence['error_propagation_intervals']:
  k=(e['GT_OFFLINE_ONLY'],e['view']);result[k]=max(result[k],e['observed_wrong_frames'])
 return result
def compare(event,base,probe):
 assert base['starting_native_state_fp']==probe['starting_native_state_fp']
 assert base['actual_mutated_state'] and probe['actual_mutated_state'] and base['first_scores_exact'] and probe['first_scores_exact']
 boundary=tuple(event['event_key'][1:3]);target=tuple(event['event_key'][1:])
 bt=rows(base['owner_trace']['path']);pt=rows(probe['owner_trace']['path']);checks={};all_harms=set();real=False;strict=True
 for h in [16,32,64]:
  b=indexed(bt,boundary,h);p=indexed(pt,boundary,h);assert b.keys()==p.keys()
  transitions=collections.Counter();introduced=collections.Counter();changed_assessed=collections.Counter()
  for k,r in b.items():
   s=p[k];assert r['GT_OFFLINE_ONLY']==s['GT_OFFLINE_ONLY']
   if r['wrong_owner'] and s['owner_correct']:transitions['N01']+=1
   if r['owner_correct'] and s['wrong_owner']:transitions['N10']+=1
   if r['owner_correct'] and not s['owner_assessed']:transitions['CORRECT_to_UNKNOWN']+=1
   if r['wrong_owner'] and not s['owner_assessed']:transitions['WRONG_to_UNKNOWN']+=1
   if not r['owner_assessed'] and s['owner_correct']:transitions['UNKNOWN_to_CORRECT']+=1
   if not r['owner_assessed'] and s['wrong_owner']:transitions['UNKNOWN_to_WRONG']+=1
   if r['identity']!=s['identity']:changed_assessed['known' if s['owner_assessed'] else 'UNKNOWN']+=1
   for f in ['new_cross_GT_gallery_mix','false_split_birth_anchored','false_split_birth']:
    introduced[f]+=int(s[f] and not r[f])
  target_repair=event['false_split_with_anchored_support'] and b[target]['action']=='START_NEW' and p[target]['owner_correct'] and p[target]['action']!='START_NEW'
  real|=transitions['N01']>0 or target_repair
  bh=base['H'][str(h)];ph=probe['H'][str(h)];bc=bh['identity_consequences'];pc=ph['identity_consequences']
  deltas={f:pc['counts'].get(f,0)-bc['counts'].get(f,0) for f in set(bc['counts'])|set(pc['counts'])}
  for f in ['UNKNOWN_current','UNKNOWN_owner','wrong_owner_observed_frames','wrong_owner_episode_max_frames']:deltas[f]=pc[f]-bc[f]
  bm=episode_max(bc);pm=episode_max(pc)
  longer=[dict(GT_OFFLINE_ONLY=k[0],view=k[1],BASE=bm.get(k,0),PROBE=n) for k,n in pm.items() if n>bm.get(k,0)]
  clear=None if bh['future_CLEAR_IDSW'] is None or ph['future_CLEAR_IDSW'] is None else ph['future_CLEAR_IDSW']-bh['future_CLEAR_IDSW']
  strict&=clear is not None
  harms=[]
  if transitions['N10']:harms.append('ANTI_CORRECTION')
  if transitions['CORRECT_to_UNKNOWN']:harms.append('CORRECT_COVERAGE_LOSS')
  if introduced['new_cross_GT_gallery_mix']:harms.append('INTRODUCED_MIX')
  if introduced['false_split_birth_anchored'] or introduced['false_split_birth']:harms.append('INTRODUCED_FALSE_SPLIT')
  if deltas['wrong_owner_observed_frames']>0:harms.append('MORE_WRONG_OBSERVED_FRAMES')
  if longer:harms.append('LONGER_GT_CAMERA_WRONG_EPISODE')
  if clear is not None and clear>0:harms.append('MORE_STRICT_CLEAR_IDSW')
  all_harms.update(harms)
  checks[str(h)]=dict(transitions=dict(transitions),introduced_harm_rows=dict(introduced),changed_ID_rows=dict(changed_assessed),anchored_target_split_repaired=target_repair,consequence_deltas=deltas,delta_strict_CLEAR_IDSW=clear,longer_GT_camera_episodes=longer,harm_reasons=harms,strict_CLEAR_evaluable=clear is not None)
 audit=probe['constraint'];other_accepted=[r for r in audit['changed_rows'] if r!=audit['row'] and audit['original_columns'][r]<len(audit['refs'])]
 return dict(event_key=event['event_key'],selection_index=event['selection_index'],split=event['split'],category=event['sampled_stratum'],variant=probe['variant'],reference=probe['reference'],BASE=ref(Path(base['owner_trace']['path']).parent/'RESULT.json'),PROBE=ref(Path(probe['owner_trace']['path']).parent/'RESULT.json'),H=checks,real_correction=bool(real),harm_reasons=sorted(all_harms),strict_CLEAR_all_H_evaluable=strict,safe_correction=bool(real and not all_harms and strict),owner_only_safe_diagnostic=bool(real and not all_harms),constraint=audit,other_original_MATCH_rows_reassigned=other_accepted)

def coverage(events):
 out=[]
 for label in ['AVAILABLE_ERROR','UNAVAILABLE_ERROR','CORRECT_CONTROL']:
  items=[e for e in events if e['sampled_stratum']==label];known=[e for e in items if e['correct_owner_refs_OFFLINE']]
  out.append(dict(stratum=label,groups=len(items),correct_owner_in_active_MATCH=sum(bool(e['correct_owner_refs_OFFLINE']) for e in items),correct_owner_in_GT_free_alternatives=sum(bool(e['alternative_owner_coverage_OFFLINE']) for e in items),numeric_Recall_at_K={str(k):dict(numerator=sum(bool(e['numeric_Recall_at_K'].get(str(k))) for e in known),denominator=len(known),value=sum(bool(e['numeric_Recall_at_K'].get(str(k))) for e in known)/len(known) if known else None) for k in [1,4,8,16,32]},unexecuted_action_values=None))
 return out

def main():
 protect();events=read(OUT/'selection/INTERVENTION_EVENTS.json');sampled=read(OUT/'selection/SAMPLED_EVENTS.json');qualification=read(REPORTS/'INDEPENDENCE_QUALIFICATION_PROTOCOL.json');pairs=[];sources=[];all_branches=[]
 for video in TRAIN:
  path=OUT/('interventions_v2' if video==16 else 'interventions')/f'video{video:02d}/RESULT.json';vr=read(path);assert vr['status']=='COMPLETE';sources.append(ref(path));all_branches+=vr['branches']
 lookup={}
 for source in all_branches:
  assert sha(source['path'])==source['SHA256'];r=read(source['path']);lookup[r['event_key'][0],r['event_key'][1],r['event_key'][2],r['event_key'][3],r['variant']]=r
 assert len(lookup)==sum(1+len(e['alternatives_GT_free']) for e in events)==73
 for e in events:
  key=tuple(e['event_key']);base=lookup[key+('BASE',)]
  for n,refid in enumerate(e['alternatives_GT_free']):
   probe=lookup[key+(f'ALT{n}',)];assert probe['reference']==refid;pairs.append(compare(e,base,probe))
 group_rows=[]
 for e in events:
  ps=[p for p in pairs if p['selection_index']==e['selection_index']]
  group_rows.append(dict(selection_index=e['selection_index'],event_key=e['event_key'],split=e['split'],category=e['sampled_stratum'],correct_owner_in_alternatives=e['alternative_owner_coverage_OFFLINE'],safe_variants=[p['variant'] for p in ps if p['safe_correction']],owner_only_safe_variants=[p['variant'] for p in ps if p['owner_only_safe_diagnostic']],harmful_variants=[p['variant'] for p in ps if p['harm_reasons']],real_correction_variants=[p['variant'] for p in ps if p['real_correction']]))
 safe={g['selection_index'] for g in group_rows if g['safe_variants']};qualified_train=[];qualified_reserved=[];units=[]
 for component,members in qualification['qualification_components'].items():
  splits={events[i]['split'] for i in members};has_safe=bool(set(members)&safe)
  row=dict(component=component,members=members,mixed_splits=len(splits)>1,safe_event_members=sorted(set(members)&safe),qualified_train=has_safe and splits=={'TRAIN'},qualified_reserved=has_safe and splits=={'RESERVED'});units.append(row)
  if row['qualified_train']:qualified_train.append(component)
  if row['qualified_reserved']:qualified_reserved.append(component)
 summary=dict(groups=16,baselines=16,alternatives=len(pairs),total_complete_native_branches=73,safe_temporal_representatives=len(safe),safe_action_variants=sum(p['safe_correction'] for p in pairs),real_correction_action_variants=sum(p['real_correction'] for p in pairs),harmful_action_variants=sum(bool(p['harm_reasons']) for p in pairs),owner_only_safe_action_variants=sum(p['owner_only_safe_diagnostic'] for p in pairs),strict_CLEAR_unevaluable_action_variants=sum(not p['strict_CLEAR_all_H_evaluable'] for p in pairs),independent_qualification_units=len(units),independent_safe_TRAIN_groups=len(qualified_train),independent_safe_RESERVED_groups=len(qualified_reserved),required_TRAIN=8,required_RESERVED=3,learning_gate='NO_GO_INDEPENDENCE_OR_SAFETY',new_training='NOT_RUN',new_policy_performance=None,full_cold_online_new_policy='NOT_RUN',new_HOTA_AssA_IDF1_MOTA=None)
 save(REPORTS/'NATIVE_PAIRED_OWNER_OUTCOMES.json',dict(status='COMPLETE_BOUNDED_DIAGNOSTIC',binding=binding(),sources=sources,rules=ref(REPORTS/'OUTCOME_QUALIFICATION_RULES.json'),summary=summary,groups=group_rows,qualification_units=units,pairs=pairs,scientific_limitations=['counterfactual oracle labels are diagnostics, not deployable policy performance','overlapping groups/actions/H are correlated; no significance test or independent sample inflation','UNKNOWN is not correction; owner labels are immutable past-only estimates, not perfect identifiability','video16 duplicate GT yields null strict CLEAR when affected prefix/future is included','no additional second MATCH scoring validation exists in frozen native adapter','ACTIVE_MATCH_UNAVAILABLE does not prove owner absent from Bank or all history']))
 save(REPORTS/'CANDIDATE_ACTION_COVERAGE.json',dict(status='COMPLETE',sampled_events=ref(OUT/'selection/SAMPLED_EVENTS.json'),intervention_events=ref(OUT/'selection/INTERVENTION_EVENTS.json'),sampled64=coverage(sampled),intervened16=coverage(events),availability_scope='legal active MATCH pool; actual REACT queried pool recorded separately; counterfactual Bank pool for accepted MATCH not observed',unknown_correct_owner_availability='UNKNOWN/null, not missing-owner proof',no_top16_truncation=True,unexecuted_action_values=None,GT_only_offline=True))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
