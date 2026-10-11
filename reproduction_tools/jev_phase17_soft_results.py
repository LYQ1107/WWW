"""Prospective prefix safety gates; unknown owner is never counted as correction."""
from jev_phase17_common import *
import collections,gzip

def main():
 protect();result={}
 for rule in ['E1','E2']:
  harms=[];benefit=set();cases=[];changed=0
  for video in TRAIN:
   base=read(OUT/'factorial/prefix/B0'/f'video{video:02d}/RESULT.json');actual=read(OUT/'soft/prefix'/rule/f'video{video:02d}/RESULT.json')
   for b,r in zip(base['cases'],actual['cases']):
    assert b['prefix']==r['prefix'];assert b['startup']==r['startup'],(rule,video,'different actual initial state');entry=b['prefix'];strict=video!=14;unsafe=False;improved=False;hs={};key=entry['key'];row=entry['row'];changed+=r['diagnostic_counts']['same_state_changed_assignments']
    assert r['diagnostic_counts'].get('Q4_changed_queries',0)==0
    with gzip.open(b['owner_trace']['path'],'rt') as f:
     target=next(json.loads(line) for line in f if json.loads(line)['key']==key[1:] and json.loads(line)['row']==row)
    for h in ['8','16','32']:
     a=r['H'][h]['identity_consequences'];old=b['H'][h]['identity_consequences'];d={name:a['counts'].get(name,0)-old['counts'].get(name,0) for name in ['new_cross_GT_gallery_mix','false_split_birth']};d['wrong_owner_observed_frames']=a['wrong_owner_observed_frames']-old['wrong_owner_observed_frames']
     def pergt(x):
      p={}
      for e in x['error_propagation_intervals']:
       k=e['GT_OFFLINE_ONLY'],e['view'];p[k]=max(p.get(k,0),e['observed_wrong_frames'])
      return p
     now=pergt(a);before=pergt(old);longer=[dict(GT_OFFLINE_ONLY=k[0],view=k[1],delta=v-before.get(k,0)) for k,v in now.items() if v>before.get(k,0)]
     bad={k:v for k,v in d.items() if v>0}
     if longer:bad['longer_per_GT_camera_wrong_episode']=longer
     unsafe|=bool(bad);improved|=d['wrong_owner_observed_frames']<0 or d['new_cross_GT_gallery_mix']<0
     if bad and strict:harms.append(dict(video=video,key=key,row=row,H=int(h),harm=bad))
     hs[h]=dict(delta=d,actual_CLEAR_IDSW=r['H'][h]['future_CLEAR_IDSW'],original_CLEAR_IDSW=b['H'][h]['future_CLEAR_IDSW'],actual_UNKNOWN=a['UNKNOWN_owner'],original_UNKNOWN=old['UNKNOWN_owner'],harm=bad)
    if strict and improved and not unsafe and target['GT_OFFLINE_ONLY'] is not None:benefit.add((video,target['GT_OFFLINE_ONLY'],key[2],key[1]//64))
    cases.append(dict(video=video,key=key,row=row,strict=strict,same_start=True,H=hs,source=ref(Path(r['raw_predictions']['path']).parent/'RESULT.json')))
  contract=read(OUT/'soft/contract'/rule/'video17/RESULT.json');assert contract['diagnostic_counts'].get('Q4_changed_queries',0)==0
  result[rule]=dict(prefix_gate='NO_GO_PREFIX_SAFETY' if harms else 'PASS_PREFIX_SAFETY',full_cold_qualified=not harms,qualified_safe_training_groups=None,local_H32_safe_benefit_groups=len(benefit),harmful_prefix_H_checks=len(harms),harmful_checks=harms,native_contract='PASS_RAW_LATEST_Q4_MASK_META_LEGAL_INVARIANTS',same_state_changed_assignments=changed,completed_prefix_branches=24,strict_prefixes=18,diagnostic_video14=6,cases=cases,cold_runs=dict(status='NOT_RUN',metrics=None,reason='unsafe prefix' if harms else 'qualified cold runs pending'))
 save(REPORTS/'SOFT_MEMORY_CAUSAL_RESULTS.json',dict(status='PREFIX_COMPLETE',binding=binding(),protocol=ref(REPORTS/'SOFT_MEMORY_PROTOCOL.json'),rules=result,UNKNOWN_not_correction=True,paired_native_mutated_state=True,scope='one version per rule; H8/16/32 overlapping checks not independent, raw perception/checkpoint/native assignment preserved',no_tuning_after_failure=True))
 print('SOFT_GATES',[(k,v['prefix_gate'],v['local_H32_safe_benefit_groups'],v['harmful_prefix_H_checks']) for k,v in result.items()],flush=True)
if __name__=='__main__':main()
