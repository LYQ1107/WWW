"""Observed external birth/candidate growth; no outcome-dependent pruning or tuning."""
import collections,gzip
from jev_phase14_common import *

def main():
    protect();queue=read(OUT/'external_eval_v1/completion_queue_v1/RESULT.json');assert queue['status']=='COMPLETE';cases=[]
    for item in queue['done']:
        path=Path(item['result']);r=read(path);file=path.parent/'COMMITS.jsonl.gz';counts=collections.Counter();window=collections.deque();hist=collections.Counter();peak=0;maximum_id_count=0;trace_points=[]
        with gzip.open(file,'rt') as h:
            for line in h:
                e=json.loads(line);frame=e['key'][1];maximum_id_count=max(maximum_id_count,e['id_count'])
                for action in e['events']:counts[action['action']]+=1
                window.append((frame,e['ids']));hist.update(e['ids'])
                while window and window[0][0]<frame-39:
                    f,ids=window.popleft()
                    for t in ids:
                        hist[t]-=1
                        if not hist[t]:del hist[t]
                peak=max(peak,len(hist))
                if frame%32==0 and e['key'][2]==1:trace_points.append(dict(frame=frame,id_count=e['id_count'],IDs_committed_over_recent40scene_frames=len(hist)))
        assert sha(file)==r['commits_SHA256']
        cases.append(dict(variant=r['variant'],seed=r['seed'],historical=r['historical'],actions=dict(counts),maximum_native_id_count=maximum_id_count,
            peak_IDs_committed_over_recent40scene_frames=peak,progression=trace_points,cached_run_seconds_including_GT_TrackEval_MATLAB=r['seconds'],
            measured_full_FPS=None,scope='observed trailing40scene-frame committed-ID union, not a direct measurement of each NN legal candidate-set K; cached+journal+evaluation wall time is not live FPS',
            result=dict(path=str(path),SHA256=sha(path)),commits=dict(path=str(file),SHA256=sha(file))))
    save(REPORTS/'EXTERNAL_RUNTIME_DIAGNOSTICS.json',dict(status='COMPLETE',binding=binding(),cases=cases,
        inference='birth growth increases native history/bank work and the Set option self-attention path has quadratic cost in lawful option count; this external case has no separately measured component attribution',
        learned_policy_and_legal_candidate_sets_preserved=True,no_forced_reset_pruning_or_threshold_changes=True,progressing_slow_case_is_not_a_stalled_frontend=True))
    print('PHASE14_EXTERNAL_RUNTIME_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
