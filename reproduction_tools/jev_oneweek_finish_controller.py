"""Finite evidence completion controller, preserving a scoped scientific NO_GO.

This is an experiment dependency controller, not a recurring user notification.
It never trains/selects new versions or changes old MASTER safety gates.
"""
import traceback
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
    run_job('formal_identity_final_snapshot',[PYTHON,str(ROOT/'reproduction_tools/jev_oneweek_identity_matrix.py')])
    # All immutable native and true timing inputs exist. Readiness never means
    # the model won, and the original MASTER remains incomplete.
    ready=dict(status='COMPLETE_FROZEN_SPRINT_EVIDENCE',required_files=[ref(p) for p in files],
               main_science_claim='NOT_ESTABLISHED',algorithm_major_revisions_used=1,
               original_MASTER_goal_achieved=False,official_TEST=False,Full24=False)
    save(root/'RESULT.json',ready);save(OUT/'FINAL_AUDIT_READY.json',ready)
    publish('Freeze complete JEVTrack bounded research evidence with honest scientific contribution judgment')
    print('ONEWEEK_FINITE_EVIDENCE_COMPLETE_ORIGINAL_MASTER_UNFINISHED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        failure('finish_controller',traceback.format_exc());raise
