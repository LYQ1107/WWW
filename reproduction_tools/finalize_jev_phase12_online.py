"""Actual pooled TrackEval, both strict causal and canonical filtered outputs."""
import argparse,time,collections,fcntl
import numpy as np
from jev_phase12_common import *
from run_jev_phase10_closed_loop import metrics

def pool_case(case):
    # Multiple reviewers may request a partial report while the watcher runs.
    # Lock a whole deterministic formatter/evaluator transaction, not files
    # individually, to avoid racing its non-atomic directory preparation.
    root=OUT/'pooled_validation_locks';root.mkdir(exist_ok=True)
    with (root/(case['name']+'.lock')).open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return None
        return _pool_case(case)

def _pool_case(case):
    name=case['name'];runs=[]
    for video in VAL:
        p=OUT/'validation_closed_loop_v1'/name/f'video{video:02d}/RESULT.json'
        if not p.exists():return None
        r=json.loads(p.read_text());assert r['status']=='COMPLETE' and r['case']==case;assert r['actual_mutated_state_online'];runs.append(r)
    out=OUT/'pooled_validation_v1'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():return json.loads((out/'RESULT.json').read_text())
    pooled={};artifacts={}
    for scope,field in [('strict_online','strict_predictions'),('canonical_GMT_filtered','canonical_predictions')]:
        predictions=[]
        for run in runs:
            p=run[field]['path'];assert sha(p)==run[field]['SHA256'];predictions.extend(json.loads(Path(p).read_text()))
        path=out/(scope+'_PREDICTIONS.json');save(path,predictions);value,evaluated=metrics(path,VAL,out/scope);pooled[scope]=value
        artifacts[scope]={'combined_metrics_path':str(evaluated/'metrics.json'),'SHA256':sha(evaluated/'metrics.json'),'prediction_SHA256':sha(path)}
    candidate=collections.Counter();identities=collections.Counter();cross=[];latency=[];stages=[]
    for run in runs:
        candidate.update(run['online_candidate_lifecycle_audit']['counts']);identities.update({k:v for k,v in run['identity_summary'].items() if isinstance(v,(int,float))});cross.append(run['cross_camera'])
        p=OUT/'validation_closed_loop_v1'/name/f"video{run['video']:02d}"/'LATENCY_SAMPLES.json';d=json.loads(p.read_text());latency.extend(d['policy'][10:]);stages.extend(d['typed_MATCH'][10:])
    def quantile(data,key):
        return dict(zip(['p50','p95','max'],map(float,np.quantile([d[key] for d in data],[.5,.95,1])))) if data else None
    diagnostics={'counts':dict(candidate),'identity_global_counts':dict(identities),'wrong_identity_duration_camera_frames_sum':sum(r['online_candidate_lifecycle_audit']['wrong_ID_duration_total_camera_frames'] for r in runs),'contaminated_gallery_tracks_sum':sum(r['online_candidate_lifecycle_audit']['contaminated_gallery_tracks'] for r in runs),'cross_camera_per_video':cross,'candidate_accuracy_lower_bound':candidate['confirmed_correct_observations']/max(1,sum(candidate[x] for x in ['confirmed_correct_observations','confirmed_wrong_observations','GT_unknown','anchor_unknown'])),'candidate_accuracy_upper_bound':1-candidate['confirmed_wrong_observations']/max(1,sum(candidate[x] for x in ['confirmed_correct_observations','confirmed_wrong_observations','GT_unknown','anchor_unknown'])),'candidate_rank1':candidate['candidate_rank1_hits']/max(1,candidate['ranking_evaluable_rows']),'candidate_MRR':candidate['candidate_MRR_sum']/max(1,candidate['ranking_evaluable_rows']),'candidate_full_recall':candidate['available_correct_rows']/max(1,candidate['known_GT_candidate_rows']),'scope':'permanent first-two prefix anchors, UNKNOWN neither correct nor wrong; bounds are committed observation certificates, ranking separate; camera-frame propagation, not FPS-converted seconds'}
    result={'status':'COMPLETE','case':case,'pooled_metrics':pooled,'pooled_metric_artifacts':artifacts,'pooling':'TrackEval COMBINED_SEQ over all3videos; HOTA/AssA average over19alpha thresholds, not average of3sequence HOTA values','diagnostics':diagnostics,'per_video':[{k:v for k,v in r.items() if k not in ['identity_summary','cross_camera','online_candidate_lifecycle_audit']} for r in runs],'latency':{'policy_assignment_ms':quantile(latency,'policy_assignment_ms'),'extra_vs_GMT_policy_ms':quantile(latency,'extra_policy_assignment_ms'),'native_token_build_and_match_ms':quantile(stages,'total_ms'),'peak_VRAM_MiB':quantile(latency,'peak_VRAM_MiB'),'scope':'real actors during concurrent GPU queue; isolated native-prefix benchmark reported separately'},'paired_H32_regret_sum':sum(r['paired_H32_causal_audit']['H32_regret_against_executed_Fixed_alternative'] for r in runs),'counts':dict(sum((collections.Counter(r['counts']) for r in runs),collections.Counter()))}
    save(out/'RESULT.json',result);print('ACTUAL_POOLED_CASE_COMPLETE',name,pooled['strict_online'],flush=True);return result

def main(partial):
    protect();protocol=json.loads((REPORTS/'ONLINE_PROTOCOL.json').read_text());cases=[]
    for case in protocol['cases']:
        value=pool_case(case)
        if value is not None:cases.append(value)
    complete=len(cases)==len(protocol['cases']);save(OUT/'POOLED_PROGRESS.json',{'status':'COMPLETE' if complete else 'RUNNING','complete_cases':len(cases),'total_cases':len(protocol['cases']),'case_names':[r['case']['name'] for r in cases]})
    if not complete:
        assert partial,'do not publish partial results as completed validation';return False
    byname={r['case']['name']:r for r in cases};offline=json.loads((REPORTS/'MATCH_OFFLINE_VALIDATION.json').read_text());training=json.loads((REPORTS/'MATCH_TRAINING_PROTOCOL.json').read_text());groups={}
    for model in training['models']+['full_no_risk']:
        records=[byname[f'{model}_s{seed}'] for seed in training['seeds']]
        groups[model]={'seeds':training['seeds'],'pooled_metric_mean':{scope:{m:float(np.mean([r['pooled_metrics'][scope][m] for r in records])) for m in records[0]['pooled_metrics'][scope]} for scope in ['strict_online','canonical_GMT_filtered']},'seed_metric_range':{scope:{m:[float(min(r['pooled_metrics'][scope][m] for r in records)),float(max(r['pooled_metrics'][scope][m] for r in records))] for m in records[0]['pooled_metrics'][scope]} for scope in ['strict_online','canonical_GMT_filtered']}}
    report={'status':'COMPLETE','protocol_SHA256':sha(REPORTS/'ONLINE_PROTOCOL.json'),'videos':VAL,'native_video_actors':len(cases)*3,'cases':cases,'three_seed_groups':groups,'offline_validation':{'path':'reports/JEV_PHASE12/MATCH_OFFLINE_VALIDATION.json','SHA256':sha(REPORTS/'MATCH_OFFLINE_VALIDATION.json')},'primary':'strict causal committed stream; future complete-video length filter only separate canonical benchmark','data_scopes':'3 development videos and partial native labels; no heldout confirmatory claim','architecture_weights_or_temperature_retuned_after_MOT':False,'heldout':'SEALED','Full24':False,'official_TEST':False}
    save(REPORTS/'MATCH_VALIDATION_RESULTS.json',report)
    ablation=json.loads((REPORTS/'ARCHITECTURE_ABLATION.json').read_text());ablation['online_three_seed_groups']=groups;ablation['no_risk']='same frozen Full weights and temperature; real mutated-state policy intervention';save(REPORTS/'ARCHITECTURE_ABLATION.json',ablation)
    isolated_path=OUT/'isolated_latency_v1/RESULT.json';latency={'status':'COMPLETE_ONLINE_ISOLATED_PENDING','actual_online_per_case':{r['case']['name']:r['latency'] for r in cases},'target_extra_policy_assignment_p95_ms':10.,'full_state_sharing':'once per camera payload, shared across all simultaneous MATCH questions','cross_stage_cache_hits':0,'MEMORY_REACTIVATION':'untrained native rule fallback; no trained neural execution latency claim','frozen_backbone_recomputed':False}
    if isolated_path.exists():
        isolated=json.loads(isolated_path.read_text());latency.update(status='COMPLETE',isolated_native_prefix_benchmark={k:v for k,v in isolated.items() if k!='samples'},isolated_raw_samples_SHA256=sha(isolated_path))
    save(REPORTS/'ONLINE_LATENCY.json',latency);return True
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        finished=main(a.partial or a.watch)
        if not a.watch or finished:break
        time.sleep(30)
