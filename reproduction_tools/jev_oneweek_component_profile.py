"""Separate synchronised component profile; never report its FPS as primary FPS.

Nested timings are inclusive. Gallery/Bank/ID update time is the Stage2 residual
after history, policy and native solver; it includes ordinary Python bookkeeping.
"""
import argparse
import collections
import numpy as np
import torch
from jev_oneweek_common import *
from jev_oneweek_online import policy


def main(arm):
    protect();assert arm in ['jev','parent','mlp','attention','no_shared_recompute']
    from jev_phase13_runtime import build_tracker,cache_inputs,run
    from jev_phase14_latency import prune
    from gtr.modeling.jev_oneweek.native import attach_jevtrack
    import gtr.modeling.jev_stage2.native as native
    import importlib
    proposals=importlib.import_module('gtr.modeling.jev_oneweek.plans')
    import jev_oneweek_online as controls
    from PIL import Image
    samples=collections.defaultdict(list)
    def timed(name,function):
        def call(*args,**kwargs):
            torch.cuda.synchronize();begin=time.perf_counter()
            result=function(*args,**kwargs)
            torch.cuda.synchronize();samples[name].append((time.perf_counter()-begin)*1000)
            return result
        return call
    def wrap(obj,attr,name):setattr(obj,attr,timed(name,getattr(obj,attr)))
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');assert visible and visible.isdigit()
    uuid=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    def uncontended():
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
        assert not [r for r in rows if r.strip().startswith(uuid+',') and int(r.split(',')[1])!=os.getpid()]
    uncontended();seed_all();actor,checkpoint=policy(arm,20261009,'formal')
    model=build_tracker(17,actor,react_learned=False,live=True);prune(model)
    wrap(native,'lawful_choice','native_Hungarian')
    wrap(proposals,'lawful_choice','proposal_Hungarian')
    wrap(controls,'lawful_choice','control_Hungarian')
    ex=attach_jevtrack(model)
    for obj,attr,name in [(model,'inference','Stage1_including_image_IO'),
                          (model,'preprocess_image','image_preprocess'),
                          (model.backbone,'forward','backbone'),
                          (model.proposal_generator,'forward','detector'),
                          (model.roi_heads.asso_pooler,'forward','VFCE_ROI_pool'),
                          (model.roi_heads.asso_head,'forward','VFCE_head'),
                          (Image,'open','image_open'),(Image.Image,'convert','image_decode_and_colour_convert'),
                          (ex,'batch','history_evidence_build'),(actor,'forward','policy_total')]:wrap(obj,attr,name)
    if hasattr(actor,'identity'):wrap(actor.identity.core,'encode_state','shared_state_encode')
    if hasattr(actor,'model') and hasattr(actor.model,'identity'):wrap(actor.model.identity.core,'encode_state','shared_state_encode')
    if hasattr(actor,'readers'):
        for index,reader in enumerate(actor.readers):wrap(reader,'forward','cross_reader' if index==0 else 'stability_reader')
    if hasattr(actor,'score_plans'):wrap(actor,'score_plans','joint_plan_evaluation')
    if hasattr(actor,'details'):
        # plan generation is a module-level function imported by this method.
        method=actor.details.__func__
        if 'plans' in method.__globals__:method.__globals__['plans']=timed('proposal_generation_including_Hungarian',method.__globals__['plans'])
    values,frames,_=cache_inputs(17);folder=OUT/'component_profile_v1'/arm
    folder.mkdir(parents=True,exist_ok=False)
    with torch.no_grad():raw,_=run(model,values,frames)
    torch.cuda.synchronize();uncontended()
    def summary(values):return dict(calls=len(values),sum_ms=float(sum(values)),p50_ms=float(np.median(values)) if values else None,p95_ms=float(np.quantile(values,.95)) if values else None)
    stage2=[v['total_stage2_ms'] for v in ex.latency]
    residual=sum(stage2)-sum(samples['history_evidence_build'])-sum(samples['policy_total'])-sum(samples['native_Hungarian'])
    # Stage1-to-cache parity is checked after all timed inference, with no GT
    # participating in a score/decision. Labels only supply image metadata.
    from gtr.modeling.jev_native_state import fingerprint
    from run_jev_phase10_closed_loop import raw_predictions
    from jev_phase13_common import ANNOTATIONS
    images={(i['frame_id']-1,i['view_id']-1):i for i in read(ANNOTATIONS)['images'] if i['video_id']==17}
    predictions=raw_predictions(raw,images);save(folder/'RAW_PREDICTIONS.json',predictions)
    old=read(OUT/'online_v1/formal'/('jev' if arm=='no_shared_recompute' else arm)/'seed20261009/video17/RESULT.json')
    expected=read(old['raw_predictions']['path'])
    parity=predictions==expected
    ID_parity=len(predictions)==len(expected) and all(all(a[k]==b[k] for k in ['image_id','category_id','track_id']) for a,b in zip(predictions,expected))
    max_detection_delta=max([abs(float(a[k])-float(b[k])) for a,b in zip(predictions,expected) for k in ['score']]+
                            [abs(float(u)-float(v)) for a,b in zip(predictions,expected) for u,v in zip(a['bbox'],b['bbox'])]+[0.]) if len(predictions)==len(expected) else None
    result=dict(status='COMPLETE',binding=binding('Instrumented complete real-image component profile'),arm=arm,
                checkpoint=checkpoint,scene_frames=frames,camera_payloads=2*frames,GPU_UUID=uuid,
                components={key:summary(v) for key,v in samples.items()},Stage2=summary(stage2),
                Gallery_Bank_ID_updates_and_Python_residual_ms=residual,
                Stage2_sum_exceeds_instrumented_child_components=residual>=0,
                gallery_residual_is_not_a_pure_memory_kernel=True,nested_components_not_added_as_disjoint_costs=True,
                primary_FPS=None,instrumentation_cuda_sync_changes_execution_cost=True,
                raw_prediction_exact_parity_with_frozen_perception_native_run=parity,
                native_ID_sequence_matches_cached_perception_run=ID_parity,
                maximum_raw_bbox_score_abs_delta_vs_cached_perception=max_detection_delta,
                actual_prediction_sha=fingerprint(raw[:2*frames]),hardware=torch.cuda.get_device_name())
    save(folder/'RESULT.json',result);save(REPORTS/f'COMPONENT_PROFILE_{arm}.json',result)
    assert ID_parity and max_detection_delta<=1e-4,'Live frontend changed detection/ID behavior; retained attempted profile'
    print('COMPONENT_PROFILE_COMPLETE',arm,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args()
    try:main(a.arm)
    except Exception:
        import traceback
        failure('component_profile',traceback.format_exc());raise
