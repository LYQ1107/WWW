"""Finite evidence completion controller, preserving a scoped scientific NO_GO.

This is an experiment dependency controller, not a recurring user notification.
It never trains/selects new versions or changes old MASTER safety gates.
"""
import traceback
import fcntl
from jev_oneweek_common import *
from jev_oneweek_controller import run_job
from jev_oneweek_publish import publish


def expected():
    trained=['jev','threshold','linear','state_threshold','attention','deepsets','mlp','fixed_question','cross_only','stability_only','additive','no_question_reader']
    controls=['parent','cosine','fixed_threshold','fixed_fusion','extra_hungarian','XIII_fixed_reference','XV_V3_reference',
              'base_single','base_repeat','base_rule','edge_single','edge_repeat','edge_rule','no_cross_evidence','no_stability_evidence']
    single=['cosine','XV_V3_reference','base_single','base_repeat','base_rule','edge_single','edge_repeat','edge_rule']
    files=[]
    for arm in trained+controls:
        for seed in [20261009] if arm in single else SEEDS:
            files.append(OUT/'matlab_v1/formal'/arm/f'seed{seed}/17_18_19/RESULT.json')
            for video in DEV:files.append(OUT/'online_v1/formal'/arm/f'seed{seed}/video{video:02d}/RESULT.json')
            if arm in trained:files.append(OUT/'training_v1'/arm/f'seed{seed}/formal/RESULT.json')
    for arm in ['parent','cosine','mlp','attention','jev','deepsets','fixed_question','no_shared_recompute','XIII_fixed_reference','XV_V3_reference','original_GMT']:
        files.append(OUT/'full_video_efficiency_v1/formal'/arm/'seed20261009/video17/RESULT.json')
    for arm in ['parent','jev','mlp','attention','no_shared_recompute']:files.append(OUT/'component_profile_v1'/arm/'RESULT.json')
    return files


def main():
    protect();files=expected();root=OUT/'finish_controller';root.mkdir(exist_ok=True)
    while True:
        absent=[str(p) for p in files if not p.exists()]
        failed=[k for k,v in read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs'].items() if 'FAIL' in v['status']]
        save(root/'PROGRESS.json',dict(status='WAITING_FROZEN_EVIDENCE_COMPLETION',required=len(files),missing=len(absent),
             missing_paths=absent,failed_registered_jobs=failed,next_action='Await frozen native/official/efficiency queues; no new tuning'))
        if not absent:break
        if failed:raise RuntimeError('Registered experiment failure needs actual engineering recovery: '+str(failed))
        time.sleep(30)
    for p in files:assert read(p)['status']=='COMPLETE',p
    # Profile a strongest ordinary controller if it falls outside the already
    # planned primary timing methods. Selection is only for cost verification;
    # no accuracy experiment, checkpoint or DEV sequence is selected/replaced.
    from jev_oneweek_reporting import refresh
    refresh()
    control=read(REPORTS/'GO_NO_GO.json')['association']['strongest_completed_ordinary_control']
    while read(OUT/'efficiency_queue/PROGRESS.json')['status']!='COMPLETE':time.sleep(15)
    if control and not (OUT/'full_video_efficiency_v1/formal'/control/'seed20261009/video17/RESULT.json').exists():
        run_job(f'efficiency_strongest_{control}',[PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_profile_latency.py'),
                '--arm',control,'--phase','formal','--seed','20261009','--video','17'],5)
        files.append(OUT/'full_video_efficiency_v1/formal'/control/'seed20261009/video17/RESULT.json')
        publish('Record full real-image timing for the strongest complete ordinary association control')
    run_job('remaining_actual_parameter_reads',[PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_new_parameters.py'),
            '--tag','extra','--arms','linear','state_threshold','threshold','stability_only','additive','no_question_reader'],5)
    run_job('formal_identity_final_snapshot',[PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_identity_matrix.py')])
    # All immutable native and true timing inputs exist. Readiness never means
    # the model won, and the original MASTER remains incomplete.
    ready=dict(status='COMPLETE_FROZEN_SPRINT_EVIDENCE',required_files=[ref(p) for p in files],
               main_science_claim='NOT_ESTABLISHED',algorithm_major_revisions_used=1,
               original_MASTER_goal_achieved=False,official_TEST=False,Full24=False)
    save(root/'RESULT.json',ready);save(OUT/'FINAL_AUDIT_READY.json',ready)
    with (OUT/'REGISTRY.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry=read(REPORTS/'EXPERIMENT_REGISTRY.json')
        for name in ['ablations','final_evidence','full_video_efficiency','native_closed_loop','parameter_count','pilot_controls']:
            registry['jobs'][name].update(status='COMPLETE_FROZEN_SCOPE',results=ref(root/'RESULT.json'))
        registry.update(status='COMPLETE_BOUNDED_SPRINT_EVIDENCE_SCIENTIFIC_CLAIM_NOT_ESTABLISHED',
                        original_MASTER_goal_achieved=False)
        save(REPORTS/'EXPERIMENT_REGISTRY.json',registry)
    publish('Freeze complete JEVTrack bounded research evidence with honest scientific contribution judgment')
    print('ONEWEEK_FINITE_EVIDENCE_COMPLETE_ORIGINAL_MASTER_UNFINISHED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        failure('finish_controller',traceback.format_exc());raise
