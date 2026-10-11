"""Paired real prefixes; exact reference parity and independent safety counts."""
from jev_phase17_common import *
import gzip,collections

def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def main():
 protect();cases=[];totals=collections.defaultdict(collections.Counter);harm=collections.defaultdict(list);benefit=collections.defaultdict(set);original_parity=0;hard_parity=0
 for video in TRAIN:
  manifests={a:read(OUT/'factorial/prefix'/a/f'video{video:02d}/RESULT.json') for a in ['B0','B1','B2','B3']}
  for index in range(6):
   arms={a:d['cases'][index] for a,d in manifests.items()};entry=arms['B0']['prefix'];key=entry['key'];row=entry['row'];boundary=tuple(key[1:]);assert all(r['prefix']==entry for r in arms.values());assert len({r['startup']['native_state_SHA256'] for r in arms.values()})==1;assert len({r['startup']['raw_memory_SHA256'] for r in arms.values()})==1;assert len({r['startup']['RNG_SHA256'] for r in arms.values()})==1
   traces={a:rows(r['owner_trace']['path']) for a,r in arms.items()};maps={a:{(tuple(q['key']),q['row']):q for q in traces[a]} for a in traces};name=f'frame{key[1]:06d}_view{key[2]}_row{row}'
   oldA=read(XVI/'P3_factorial/A'/f'video{video:02d}'/(name+'.json'));oldB=read(XVI/'P3_factorial/B'/f'video{video:02d}'/(name+'.json'))
   for actual,old in [('B0',oldA),('B3',oldB)]:
    for ev in rows(old['trace']['path']):
     if ev['key'][0]>=key[1]+32:continue
     assert maps[actual][tuple(ev['key']),ev['row']]['identity']==ev['identity'],(actual,video,key,ev['key'],ev['row'])
   original_parity+=1;hard_parity+=1;target=maps['B0'][boundary,row];gt=target['GT_OFFLINE_ONLY'];cluster=(video,gt,key[2],key[1]//64);strict=video!=14;horizons={};unsafe={a:False for a in ['B1','B2','B3']};improved={a:False for a in unsafe}
   for h in ['8','16','32']:
    values={}
    for a,r in arms.items():
     v=r['H'][h];c=v['identity_consequences'];values[a]=dict(IDSW=v['future_CLEAR_IDSW'],mix=c['counts'].get('new_cross_GT_gallery_mix',0),wrong=c['wrong_owner_observed_frames'],wrong_episode_max=c['wrong_owner_episode_max_frames'],false_split=c['counts'].get('false_split_birth',0),correct=c['counts'].get('owner_correct',0),UNKNOWN=c['UNKNOWN_owner'])
     if strict:totals[a+'_H'+h].update(values[a])
    deltas={a:{f:values[a][f]-values['B0'][f] for f in values[a]} for a in unsafe};interaction={f:values['B3'][f]-values['B2'][f]-values['B1'][f]+values['B0'][f] for f in values['B0']};transitions={}
    for a in unsafe:
     per={}
     for e in arms[a]['H'][h]['identity_consequences']['error_propagation_intervals']:per[e['GT_OFFLINE_ONLY'],e['view']]=max(per.get((e['GT_OFFLINE_ONLY'],e['view']),0),e['observed_wrong_frames'])
     baseline={}
     for e in arms['B0']['H'][h]['identity_consequences']['error_propagation_intervals']:baseline[e['GT_OFFLINE_ONLY'],e['view']]=max(baseline.get((e['GT_OFFLINE_ONLY'],e['view']),0),e['observed_wrong_frames'])
     bad={f:v for f,v in deltas[a].items() if f in ['mix','wrong','wrong_episode_max','false_split'] and v>0};longer=[dict(GT_OFFLINE_ONLY=k[0],camera=k[1],delta=v-baseline.get(k,0)) for k,v in per.items() if v>baseline.get(k,0)]
     if longer:bad['per_GT_camera_longer']=longer
     unsafe[a]|=bool(bad);improved[a]|=deltas[a]['wrong']<0 or deltas[a]['mix']<0
     if strict and bad:harm[a].append(dict(video=video,key=key,row=row,H=int(h),worsening=bad))
     transition=collections.Counter()
     for k,ev in maps[a].items():
      if ev['key'][0]>=key[1]+int(h) or ev['GT_OFFLINE_ONLY'] is None:continue
      old=maps['B0'][k];left='CORRECT' if old['owner_correct'] else 'WRONG' if old['wrong_owner'] else 'UNKNOWN';right='CORRECT' if ev['owner_correct'] else 'WRONG' if ev['wrong_owner'] else 'UNKNOWN';transition[left+'_TO_'+right]+=1
     transitions[a]=dict(transition)
    horizons[h]=dict(actual=values,deltas_vs_B0=deltas,interaction=interaction,owner_transitions=transitions)
   for a in unsafe:
    if strict and gt is not None and improved[a] and not unsafe[a]:benefit[a].add(cluster)
   cases.append(dict(video=video,key=key,row=row,strict=strict,H=horizons,same_start_and_RNG=True,original_ID_parity=True,hard_ID_parity=True,sources={a:ref(Path(r['raw_predictions']['path']).parent/'RESULT.json') for a,r in arms.items()}))
 report=dict(status='COMPLETE',binding=binding(),native_branches=96,prefixes=24,strict_prefixes=18,diagnostic_video14=6,H32_descriptive_overlapping_sums={a:dict(c) for a,c in totals.items() if a.endswith('_H32')},original24_prefix_IDs_exact=original_parity,hard24_prefix_IDs_exact=hard_parity,strict_harmful_prefix_H_checks={a:len(harm[a]) for a in ['B1','B2','B3']},safe_independent_benefit_clusters={a:len(benefit[a]) for a in ['B1','B2','B3']},harms=dict(harm),cases=cases,overlapping_windows_are_not_independent=True,safety_NOT_established_for_hard_memory=True,UNKNOWN_not_recovery=True)
 save(REPORTS/'PREFIX_WHO_Q4_FACTORIAL.json',report);save(REPORTS/'PHASE17_COLD_START_VS_PREFIX.json',dict(status='PREFIX_COMPLETE_COLD_METRICS_SEPARATE',prefix_report=ref(REPORTS/'PREFIX_WHO_Q4_FACTORIAL.json'),cold_report=str(REPORTS/'WHO_Q4_FACTORIAL.json'),interpretation='prefix arms retain common legacy past; cold arms filter from bootstrap; cannot combine as independent sample populations'))
 print('PREFIX_FACTORIAL_COMPLETE',report['strict_harmful_prefix_H_checks'],report['safe_independent_benefit_clusters'],flush=True)
if __name__=='__main__':main()
