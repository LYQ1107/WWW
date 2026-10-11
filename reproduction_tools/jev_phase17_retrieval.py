"""Causal identity-index retrieval diagnostics on actual predicted native IDs."""
from jev_phase17_common import *
import gzip,collections,statistics

def main():
 protect();allresults={}
 for arm in ['B0','B1','B2','B3']:
  cases=[];totals=collections.Counter();latencies=[]
  for video in DEV:
   result=read(OUT/'factorial/cold'/arm/f'video{video:02d}/RESULT.json')
   with gzip.open(result['owner_trace']['path'],'rt') as f:events=[json.loads(line) for line in f]
   index=collections.defaultdict(list);past_GT_ids=collections.defaultdict(set);counts=collections.Counter();querytimes=[];updatetimes=[]
   for ev in events:
    frame,view=ev['key'];identity=ev['identity'];gt=ev['GT_OFFLINE_ONLY'];started=time.perf_counter();returned=[r for r in index[identity] if r['view']!=view];querytimes.append((time.perf_counter()-started)*1000)
    known=[r for r in returned if r['GT_OFFLINE_ONLY'] is not None];counts['queries']+=1;counts['returned_observations']+=len(returned);counts['returned_unknown_observations']+=len(returned)-len(known)
    if gt is not None and known:
     counts['known_queries_with_past_other_view_results']+=1;correct=sum(r['GT_OFFLINE_ONLY']==gt for r in known);wrong=len(known)-correct;counts['correct_returned_observations']+=correct;counts['wrong_returned_observations']+=wrong;counts['polluted_queries']+=wrong>0;counts['queries_any_correct_cross_view']+=correct>0;counts['queries_all_returned_correct']+=wrong==0
     # Fragment entries group consecutive observations with same predicted ID, view and offline GT.
     segments=[]
     for r in known:
      if not segments or r['view']!=segments[-1]['view'] or r['GT_OFFLINE_ONLY']!=segments[-1]['GT_OFFLINE_ONLY'] or r['frame']!=segments[-1]['end']+1:segments.append(dict(view=r['view'],GT_OFFLINE_ONLY=r['GT_OFFLINE_ONLY'],end=r['frame']))
      else:segments[-1]['end']=r['frame']
     counts['returned_known_segments']+=len(segments);counts['wrong_returned_segments']+=sum(r['GT_OFFLINE_ONLY']!=gt for r in segments)
    if gt is not None:
     old=past_GT_ids[gt,1-view]
     if old:counts['known_queries_with_any_past_other_view_person']+=1;counts['cross_view_shared_predicted_ID']+=identity in old
     past_GT_ids[gt,view].add(identity)
    started=time.perf_counter();index[identity].append(dict(frame=frame,view=view,GT_OFFLINE_ONLY=gt));updatetimes.append((time.perf_counter()-started)*1000)
   c=dict(counts);c['cross_view_query_precision']=counts['correct_returned_observations']/max(1,counts['correct_returned_observations']+counts['wrong_returned_observations']);c['wrong_fragment_fraction']=counts['wrong_returned_segments']/max(1,counts['returned_known_segments']);c['cross_view_identity_continuity']=counts['cross_view_shared_predicted_ID']/max(1,counts['known_queries_with_any_past_other_view_person']);cases.append(dict(video=video,counts=c,source=result['owner_trace'],query_ms_p50=statistics.median(querytimes),query_ms_p95=sorted(querytimes)[int(.95*(len(querytimes)-1))],append_update_ms_p50=statistics.median(updatetimes)));totals.update(counts);latencies+=querytimes
  c=dict(totals);c['cross_view_query_precision']=totals['correct_returned_observations']/max(1,totals['correct_returned_observations']+totals['wrong_returned_observations']);c['wrong_fragment_fraction']=totals['wrong_returned_segments']/max(1,totals['returned_known_segments']);c['cross_view_identity_continuity']=totals['cross_view_shared_predicted_ID']/max(1,totals['known_queries_with_any_past_other_view_person']);allresults[arm]=dict(cases=cases,pooled_counts=c,query_ms_p50=statistics.median(latencies),query_ms_p95=sorted(latencies)[int(.95*(len(latencies)-1))])
 save(REPORTS/'IDENTITY_INDEX_RETRIEVAL.json',dict(status='COMPLETE_DIAGNOSTIC',binding=binding(),arms=allresults,query='current predicted GlobalID -> committed past other-camera observations under that ID, before current update',GT_usage='offline correctness only, no GT filter in index insertion or query; unknown results reported',scope='causal materialized index replay on actual native commit traces; bootstrap-only observations excluded because owner trace has no bootstrap rows; detector observed corpus only',timing='CPU standalone append-only list reference index, excludes video/tracker execution; not deployment serving benchmark',statistical_units='3 reused DEV videos; pooled counts, no mean of per-video precision',wrong_ID_duration='actual owner propagation retained in FIRST_DIVERGENCE.json',web_relevance='continuously updated multi-source video identity retrieval index reliability'))
 print('IDENTITY_RETRIEVAL_COMPLETE',{a:r['pooled_counts'] for a,r in allresults.items()},flush=True)
if __name__=='__main__':main()
