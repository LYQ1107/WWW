"""Past-only retrieval value on already-executed native branches; no new actions."""
from jev_phase20_common import *
import gzip,collections,copy


def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]


def append(index,truth,group):
 for r in group:
  tag=str(r['GT_OFFLINE_ONLY']) if r['GT_OFFLINE_ONLY'] is not None else 'UNKNOWN'
  index[r['identity'],r['key'][1]][tag]+=1
  if tag!='UNKNOWN':truth[tag,r['key'][1]]+=1


def evaluate(index,truth,future,boundary,h):
 index=copy.deepcopy(index);truth=truth.copy();counts=collections.Counter()
 groups=collections.defaultdict(list)
 for r in future:
  if boundary<=tuple(r['key']) and r['key'][0]<boundary[0]+h:groups[tuple(r['key'])].append(r)
 for key,group in sorted(groups.items()):
  for r in group:
   # Retrieval only uses opaque predicted ID and camera, never GT.
   returned=index[r['identity'],1-key[1]]
   total=sum(returned.values());unknown=returned['UNKNOWN'];known=total-unknown
   counts['queries']+=1;counts['all_queries_returned_observations']+=total
   counts['all_queries_returned_UNKNOWN']+=unknown
   gt=r['GT_OFFLINE_ONLY']
   if gt is None:counts['UNKNOWN_current_queries']+=1;continue
   tag=str(gt);correct=returned[tag];available=truth[tag,1-key[1]]
   counts['known_current_queries']+=1;counts['known_queries_returned_known']+=known
   counts['known_queries_returned_UNKNOWN']+=unknown
   counts['correct_returned']+=correct;counts['wrong_returned']+=known-correct
   counts['known_query_with_any_results']+=total>0
   counts['known_query_with_known_results']+=known>0
   counts['polluted_known_queries']+=known>correct
   counts['same_GT_past_opposite_camera_available']+=available
   counts['known_queries_with_GT_available']+=available>0
   counts['known_queries_retrieving_correct_GT']+=correct>0
  append(index,truth,group)
 def ratio(a,b):return counts[a]/counts[b] if counts[b] else None
 return dict(counts=dict(counts),
  known_return_precision=ratio('correct_returned','known_queries_returned_known'),
  known_return_pollution=ratio('wrong_returned','known_queries_returned_known'),
  known_query_known_return_coverage=ratio('known_query_with_known_results','known_current_queries'),
  same_person_observation_recall=ratio('correct_returned','same_GT_past_opposite_camera_available'),
  correct_crossview_query_coverage=ratio('known_queries_retrieving_correct_GT','known_queries_with_GT_available'))


def main():
 protect();start=time.monotonic();assert not(OUT/'RETRIEVAL_RESULT.json').exists()
 protocol=read(REPORTS/'RETRIEVAL_DIAGNOSTIC_PROTOCOL.json');s=protocol['population'];assert sha(s['path'])==s['SHA256']
 paired=read(s['path']);pairs=[];baselines={};prefixes={};sources=[]
 for pair in paired['pairs']:
  for name in ['BASE','PROBE']:
   source=pair[name];assert sha(source['path'])==source['SHA256']
  base=read(pair['BASE']['path']);probe=read(pair['PROBE']['path'])
  assert base['starting_native_state_fp']==probe['starting_native_state_fp']
  key=tuple(pair['event_key']);video=key[0];boundary=key[1:3]
  if key not in prefixes:
   c=census(video);source=c['owner_trace'];assert sha(source['path'])==source['SHA256'];sources.append(source)
   prefix=[r for r in rows(source['path']) if tuple(r['key'])<boundary]
   index=collections.defaultdict(collections.Counter);truth=collections.Counter();append(index,truth,prefix)
   prefixes[key]=index,truth
  index,truth=prefixes[key]
  histories={}
  for name,result in [('BASE',base),('PROBE',probe)]:
   source=result['owner_trace'];assert sha(source['path'])==source['SHA256'];sources.append(source)
   if name=='BASE' and key in baselines:histories[name]=baselines[key];continue
   trace=rows(source['path']);histories[name]={str(h):evaluate(index,truth,trace,boundary,h) for h in [16,32,64]}
   if name=='BASE':baselines[key]=histories[name]
  deltas={}
  for h in ['16','32','64']:
   b=histories['BASE'][h];p=histories['PROBE'][h];d={}
   for m in ['known_return_precision','known_return_pollution','known_query_known_return_coverage','same_person_observation_recall','correct_crossview_query_coverage']:
    d[m]=None if b[m] is None or p[m] is None else p[m]-b[m]
   d['correct_returned']=p['counts'].get('correct_returned',0)-b['counts'].get('correct_returned',0)
   d['wrong_returned']=p['counts'].get('wrong_returned',0)-b['counts'].get('wrong_returned',0)
   d['known_queries_returned_UNKNOWN']=p['counts'].get('known_queries_returned_UNKNOWN',0)-b['counts'].get('known_queries_returned_UNKNOWN',0)
   deltas[h]=d
  pairs.append(dict(event_key=pair['event_key'],selection_index=pair['selection_index'],variant=pair['variant'],
   split=pair['split'],owner_safe_qualification=pair['safe_correction'],
   BASE=pair['BASE'],PROBE=pair['PROBE'],H=histories,deltas=deltas))
 safe=[p for p in pairs if p['owner_safe_qualification']]
 def classify(items):
  return dict(action_variants=len(items),
   H64_precision_improved=sum((p['deltas']['64']['known_return_precision'] or 0)>0 for p in items),
   H64_precision_worsened=sum((p['deltas']['64']['known_return_precision'] or 0)<0 for p in items),
   H64_wrong_returns_increased=sum(p['deltas']['64']['wrong_returned']>0 for p in items),
   H64_wrong_returns_decreased=sum(p['deltas']['64']['wrong_returned']<0 for p in items),
   H64_UNKNOWN_returns_increased=sum(p['deltas']['64']['known_queries_returned_UNKNOWN']>0 for p in items))
 result=dict(status='COMPLETE_EXISTING_NATIVE_FUTURE_DIAGNOSTIC',binding=binding(),protocol=ref(REPORTS/'RETRIEVAL_DIAGNOSTIC_PROTOCOL.json'),
  sources=list({x['path']:x for x in sources}.values()),summary=dict(all_alternatives=classify(pairs),qualified_owner_safe=classify(safe)),pairs=pairs,
  unique_baselines=len(baselines),actual_native_branches=66,new_native_branches=0,
  seconds=time.monotonic()-start,full_video_new_policy_metrics=None,significance=None,
  previous_owner_safety_rules_unchanged=True,
  scope='Actual fixed16 prefixes and H16/32/64 mutated futures; commit-observer corpus only; f0bootstrap excluded equally.',
  latency='CPU histogram metric implementation, no serving deployment/FPS claim',
  no_pooled_across_correlated_alternative_futures=True)
 save(OUT/'RETRIEVAL_RESULT.json',result);save(REPORTS/'NATIVE_ACTION_RETRIEVAL_VALUE.json',result)
 protect();print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
