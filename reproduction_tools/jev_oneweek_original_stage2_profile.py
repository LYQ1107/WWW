"""Separate true original GMT Stage2 timing, without changing frozen core files."""
import numpy as np
import torch
from jev_oneweek_common import *
from jev_oneweek_count_params import original_model,original_live_payloads


def main():
    protect();seed_all()
    from jev_phase13_runtime import cache_inputs,run
    from run_jev_phase10_closed_loop import raw_predictions
    from jev_phase13_common import ANNOTATIONS
    visible=os.environ['CUDA_VISIBLE_DEVICES'];assert visible=='5'
    uuid=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    def clean():
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
        assert not [r for r in rows if r.strip().startswith(uuid+',') and int(r.split(',')[1])!=os.getpid()]
    clean();values,frames,_=cache_inputs(17);model,cfg=original_model();original_live_payloads(model,values,cfg)
    original=model.get_asso;start=[None];times=[];commit_depth=[0];nested_bank_association_calls=[0]
    def get_asso(*args,**kwargs):
        if start[0] is None:
            assert commit_depth[0]==0
            torch.cuda.synchronize();start[0]=time.perf_counter()
        else:
            # memory_bank invokes get_asso inside the current native commit.
            # Its cost belongs to this same Stage2 interval, not a new payload.
            assert commit_depth[0]>0
            nested_bank_association_calls[0]+=1
        return original(*args,**kwargs)
    model.get_asso=get_asso
    def wrap_commit(function):
        def commit(*args,**kwargs):
            assert start[0] is not None
            commit_depth[0]+=1
            try:value=function(*args,**kwargs)
            finally:commit_depth[0]-=1
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start[0])*1000);start[0]=None
            return value
        return commit
    model.run_first_tracker_plus=wrap_commit(model.run_first_tracker_plus)
    model.run_global_tracker_plus=wrap_commit(model.run_global_tracker_plus)
    with torch.no_grad():raw,_=run(model,values,frames)
    clean();assert len(times)==2*frames-1 and start[0] is None
    images={(i['frame_id']-1,i['view_id']-1):i for i in read(ANNOTATIONS)['images'] if i['video_id']==17}
    predictions=raw_predictions(raw,images)
    frozen=read(REPORTS/'ORIGINAL_GMT_RESTORE_VIDEO17.json')['predictions']
    exact=predictions==read(frozen['path'])
    folder=OUT/'original_stage2_profile_v1';folder.mkdir(exist_ok=False)
    save(folder/'RAW_PREDICTIONS.json',predictions)
    result=dict(status='COMPLETE',binding=binding('Actual original GMT live-image Stage2 from get_asso through original Gallery/Bank commit'),
                GPU_UUID=uuid,hardware=torch.cuda.get_device_name(),video=17,scene_frames=frames,
                camera_commits=len(times),Stage2_p50_p95_ms=list(map(float,np.quantile(times[16:],[.5,.95]))),
                nested_bank_get_asso_calls=nested_bank_association_calls[0],
                nested_bank_association_included_in_outer_commit_interval=True,
                Stage2_sum_ms=float(sum(times)),warmup_excluded_camera_payloads=16,
                includes_original_GTA_history_aggregation_threshold_assignment_Gallery_Bank=True,
                excludes_caller_instance_list_slicing_and_first_bootstrap_ID_creation=True,
                exact_live_vs_frozen_cache_raw_prediction_parity=exact,raw_predictions=ref(folder/'RAW_PREDICTIONS.json'),
                primary_FPS=None,synchronised_separate_instrumented_run_not_primary_FPS=True,
                no_GT_in_model=True,original_complete_frontend_differs_from_VFCE1024_family=True)
    save(folder/'RESULT.json',result);save(REPORTS/'ORIGINAL_GMT_STAGE2_COMPONENT_PROFILE.json',result)
    assert exact,'Actual original live-image inference differs from the frozen raw reference; attempted result retained'
    print('ORIGINAL_GMT_STAGE2_PROFILE_COMPLETE',result['Stage2_p50_p95_ms'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        failure('original_stage2_profile',traceback.format_exc());raise
