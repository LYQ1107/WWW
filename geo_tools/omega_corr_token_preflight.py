"""Official output parity and real token shapes on one preregistered physical pair."""
import json
import subprocess
import time
import numpy as np
import torch
from omega_corr_common import ROOT,RUNTIME,OMEGA_RUNTIME,save_new,sha256,provenance
from omega_modules import geometry_module


def small(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:small(x) for k,x in v.items()}
    return v


def main():
    source=provenance();assert not source['source_status'].strip()
    torch.set_num_threads(4);torch.manual_seed(20261016)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    official=OMEGA_RUNTIME/'official_vggt_omega'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()=='48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=official,text=True).strip()
    cohort=json.loads((ROOT/'manifests/omega_corr_cohort_v1.json').read_text());pair=cohort['pairs'][0]
    for i in pair['images']:assert sha256(i['path'])==i['sha256']
    module=geometry_module('omega_correspondence_encoder')
    encoder=module.OmegaCorrespondenceEncoder(official,OMEGA_RUNTIME/'weights/vggt_omega_1b_512.pt',512,layers=(4,11,17,23))
    images,mappings=encoder.prepare([i['path'] for i in pair['images']])
    def baseline():
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=encoder.amp_dtype):p=encoder.model(images.cuda())
        torch.cuda.synchronize();return p,time.perf_counter()-t
    a,t1=baseline();b,t2=baseline()
    actual=encoder.encode(images,[i['camera_id'] for i in pair['images']],pair['group'])
    comparisons={}
    for key in a:
        av,bv,cv=a[key].float(),b[key].float(),actual['predictions'][key].float()
        assert av.shape==bv.shape==cv.shape
        base=(av-bv).abs();change=(av-cv).abs()
        tolerance=3*base+1e-5+1e-4*torch.maximum(av.abs(),cv.abs())
        assert torch.isfinite(cv).all() and torch.all(change<=tolerance),key
        comparisons[key]=dict(shape=list(av.shape),baseline_repeat_max_abs=float(base.max()),hook_max_abs=float(change.max()),
                              bitwise_equal=bool(torch.equal(a[key],actual['predictions'][key])))
    h,w=actual['patch_grid'];token_shapes={str(k):list(v.shape) for k,v in actual['tokens'].items()}
    assert token_shapes=={str(k):[1,2,h*w,2048] for k in [4,11,17,23]}
    # True singleton calls and wrong-partner controls are distinct executions.
    singleton=[]
    for v in range(2):
        x=encoder.encode(images[v:v+1],[pair['images'][v]['camera_id']],pair['group'],'SINGLE_VIEW_INDEPENDENT')
        singleton.append(dict(view=v,shapes={str(k):list(z.shape) for k,z in x['tokens'].items()},seconds=x['CUDA_forward_and_hook_seconds']))
        del x
    # Small coordinate witnesses retain no images or persistent dense feature maps.
    witness={str(layer):tensor[0,:,0,:8].float().cpu().tolist() for layer,tensor in actual['tokens'].items()}
    region=geometry_module('object_patch_regions')
    mask=[dict(grid=list(region.patch_lattice(m)['grid_shape']),valid_patches=int(region.patch_lattice(m)['valid'].sum())) for m in mappings]
    output=RUNTIME/'token_preflight_v1.json'
    save_new(output,dict(status='TOKEN_INTERFACE_AND_OFFICIAL_OUTPUT_PARITY_PASS',pair_id=pair['pair_id'],
        source=source,checkpoint_sha256=sha256(OMEGA_RUNTIME/'weights/vggt_omega_1b_512.pt'),state_keys=encoder.state_keys,
        all_parameters_frozen=all(not p.requires_grad for p in encoder.model.parameters()),
        input_shape=[1,*images.shape],patch_grid=[h,w],special_tokens_excluded=17,layers=token_shapes,
        comparisons=comparisons,parity_rule='abs(hook-baseline)<=3*abs(repeat-baseline)+1e-5+1e-4*max(abs(values)); same backend, no tolerance interpreted as signal',
        baseline_calls=2,hook_joint_calls=1,true_singleton_calls=2,total_full_official_forward_calls=5,
        actual_hook_model_counter=encoder.calls_completed,singleton=singleton,witness=witness,
        mappings=[m.record() for m in mappings],masks=mask,baseline_forward_seconds=[t1,t2],
        hooked_forward_seconds=actual['CUDA_forward_and_hook_seconds'],
        CUDA_allocator_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        precision=str(encoder.amp_dtype),torch=torch.__version__,GPU=torch.cuda.get_device_name(),no_GT_input=True,
        no_F_or_pose_sweep=True,no_identity_success_claim=True))
    print('PASS real shapes',token_shapes,'official output parity',comparisons,flush=True)

if __name__=='__main__':main()
