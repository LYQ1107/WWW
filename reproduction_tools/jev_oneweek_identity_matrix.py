"""Compact all completed formal causal-ID diagnostics; detailed episodes stay local."""
import collections
import numpy as np
from jev_oneweek_common import *
from jev_oneweek_identity_analysis import analyse


def compact(a):
    durations=[e['duration_camera_frames'] for e in a['wrong_owner_episodes']]
    return dict({k:v for k,v in a.items() if k!='wrong_owner_episodes'},
                observed_error_episode_count=len(durations),
                observed_error_duration_p50_p95_camera_frames=list(map(float,np.quantile(durations,[.5,.95]))) if durations else None,
                detailed_local_analysis=ref(OUT/'online_v1'/a['phase']/a['arm']/f"seed{a['seed']}"/f"video{a['video']:02d}"/'STRICT_CAUSAL_IDENTITY_ANALYSIS.json'))


def main():
    protect();cases=[]
    arms=['jev','mlp','attention','deepsets','fixed_question','cross_only','stability_only','additive','no_question_reader',
          'threshold','state_threshold','linear','fixed_threshold','fixed_fusion','extra_hungarian',
          'XIII_fixed_reference','XV_V3_reference','no_cross_evidence','no_stability_evidence']
    for seed in SEEDS:
        for video in DEV:
            baseline_path=OUT/'online_v1/formal/parent'/f'seed{seed}'/f'video{video:02d}'/'RESULT.json'
            if not baseline_path.exists():continue
            baseline,before=analyse('parent',seed,'formal',video)
            cases.append(dict(arm='parent',seed=seed,video=video,analysis=compact(baseline)))
            for arm in arms:
                folder=OUT/'online_v1/formal'/arm/f'seed{seed}'/f'video{video:02d}'
                if not (folder/'RESULT.json').exists():continue
                actual,after=analyse(arm,seed,'formal',video)
                assert set(after)==set(before)
                count=collections.Counter()
                for key,row in after.items():
                    old=before[key]
                    if not row['known'] or not old['known']:count['UNKNOWN_comparison_rows']+=1;continue
                    assert row['gt']==old['gt']
                    a,b=int(old['correct']),int(row['correct']);count[f'N{a}{b}']+=1
                    if a==0 and b==1:
                        count['N01_new_native_ID_birth']+=row['new_native_ID']
                        count['N01_existing_native_ID']+=not row['new_native_ID']
                cases.append(dict(arm=arm,seed=seed,video=video,analysis=compact(actual),independent_cold_run_comparison=dict(count)))
            print('STRICT_IDENTITY_MATRIX_PROGRESS',seed,video,len(cases),flush=True)
            save(OUT/'identity_matrix/PROGRESS.json',dict(status='RUNNING',completed_cases=len(cases),key=[seed,video]))
    save(REPORTS/'FORMAL_STRICT_IDENTITY_MATRIX.json',dict(status='COMPLETE_SNAPSHOT_OF_AVAILABLE_FORMAL_CASES',
         binding=binding('Actual native-ID cross-camera retrieval and propagation on available full formal runs'),cases=cases,
         missing_cases_not_imputed=True,comparison_scope='N01/N10 compare independent cold-start trajectories and per-run immutable ownership; not equal-state causal effects',
         strict_GT_duplicates_UNKNOWN_official_evaluators_unmodified=True,
         detailed_episodes_not_uploaded=True,observed_duration_units='camera frames; gaps/end right-censored'))


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        failure('identity_matrix',traceback.format_exc());raise
