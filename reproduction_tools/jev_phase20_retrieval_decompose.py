"""Separate current-query repairs from subsequent index pollution on fixed branches."""
from jev_phase20_retrieval_audit import *


def rates(c):
 def ratio(a,b):return c.get(a,0)/c[b] if c.get(b,0) else None
 return dict(counts=c,known_return_precision=ratio('correct_returned','known_queries_returned_known'),
  same_person_observation_recall=ratio('correct_returned','same_GT_past_opposite_camera_available'))


def main():
 protect();assert not(OUT/'RETRIEVAL_TEMPORAL_RESULT.json').exists()
 protocol=read(REPORTS/'RETRIEVAL_TEMPORAL_PROTOCOL.json');s=protocol['source'];assert sha(s['path'])==s['SHA256']
 original=read(s['path']);prefixes={};base_first={};outputs=[]
 for p in original['pairs']:
  key=tuple(p['event_key']);boundary=key[1:3]
  if key not in prefixes:
   source=census(key[0])['owner_trace'];assert sha(source['path'])==source['SHA256']
   prefix=[r for r in rows(source['path']) if tuple(r['key'])<boundary]
   index=collections.defaultdict(collections.Counter);truth=collections.Counter();append(index,truth,prefix)
   prefixes[key]=index,truth
  index,truth=prefixes[key];future={};first={};later={}
  for name in ['BASE','PROBE']:
   s=p[name];assert sha(s['path'])==s['SHA256'];result=read(s['path']);t=result['owner_trace'];assert sha(t['path'])==t['SHA256']
   trace=rows(t['path']);future[name]={tuple(r['key'])+(r['row'],):r for r in trace}
   first[name]=evaluate(index,truth,[r for r in trace if tuple(r['key'])==boundary],boundary,1)
   later[name]={}
   for h in ['16','32','64']:
    all_counts=p['H'][name][h]['counts'];one=first[name]['counts']
    c={k:all_counts.get(k,0)-one.get(k,0) for k in set(all_counts)|set(one)}
    assert all(n>=0 for n in c.values())
    later[name][h]=rates(c)
  assert future['BASE'].keys()==future['PROBE'].keys()
  for k,b in future['BASE'].items():assert b['GT_OFFLINE_ONLY']==future['PROBE'][k]['GT_OFFLINE_ONLY']
  later_ID_changes=sum(b['identity']!=future['PROBE'][k]['identity'] for k,b in future['BASE'].items() if k[:2]!=boundary)
  delta={}
  for h in ['16','32','64']:
   b=later['BASE'][h];a=later['PROBE'][h]
   for field in ['known_current_queries','same_GT_past_opposite_camera_available']:
    assert b['counts'].get(field,0)==a['counts'].get(field,0)
   delta[h]={m:a['counts'].get(m,0)-b['counts'].get(m,0) for m in ['correct_returned','wrong_returned','known_queries_returned_UNKNOWN']}
   delta[h]['known_return_precision']=None if b['known_return_precision'] is None or a['known_return_precision'] is None else a['known_return_precision']-b['known_return_precision']
  outputs.append(dict(event_key=p['event_key'],selection_index=p['selection_index'],variant=p['variant'],
   owner_safe_qualification=p['owner_safe_qualification'],first_payload=first,later=later,later_deltas=delta,
   later_prediction_ID_rows_changed=later_ID_changes))
 safe=[p for p in outputs if p['owner_safe_qualification']]
 summary=dict(all_variants=len(outputs),qualified_owner_safe=len(safe),
  safe_H64_later_wrong_returns_decreased=sum(p['later_deltas']['64']['wrong_returned']<0 for p in safe),
  safe_H64_later_wrong_returns_increased=sum(p['later_deltas']['64']['wrong_returned']>0 for p in safe),
  safe_later_prediction_ID_change_variants=sum(p['later_prediction_ID_rows_changed']>0 for p in safe))
 result=dict(status='COMPLETE_DESCRIPTIVE_DECOMPOSITION',binding=binding(),protocol=ref(REPORTS/'RETRIEVAL_TEMPORAL_PROTOCOL.json'),
  summary=summary,pairs=outputs,new_native_actions=0,owner_qualification_unchanged=True,
  result_counts_scope='Repeated current/future service queries returning past opposite-view observations; not unique correctedGTpersons or wrong-frame duration.')
 save(OUT/'RETRIEVAL_TEMPORAL_RESULT.json',result);save(REPORTS/'NATIVE_RETRIEVAL_TEMPORAL_DECOMPOSITION.json',result)
 print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
