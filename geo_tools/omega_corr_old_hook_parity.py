"""Same-backend original old-VGGT output parity for its read-only token hook."""
import json
import torch
from omega_corr_controls import FrozenControl
from omega_corr_common import ROOT,RUNTIME,save_new,provenance


def main():
    torch.set_num_threads(4);torch.manual_seed(20261016);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    source=provenance();assert not source['source_status'].strip()
    pair=json.loads((ROOT/'manifests/omega_corr_cohort_v1.json').read_text())['pairs'][0]
    encoder=FrozenControl('old_vggt');images,_=encoder.prepare([i['path'] for i in pair['images']],[(i['original_width'],i['original_height']) for i in pair['images']])
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        a=encoder.model(images.cuda());b=encoder.model(images.cuda())
    saved={}
    def capture(module,args,output):saved['output']=output
    handle=encoder.model.register_forward_hook(capture)
    tokens,_=encoder.encode(images,[4,11,17,23]);handle.remove();c=saved['output'];comparisons={}
    for key,value in a.items():
        if not isinstance(value,torch.Tensor):continue
        x,y,z=value.float(),b[key].float(),c[key].float()
        repeat=(x-y).abs();change=(x-z).abs()
        assert torch.all(change<=3*repeat+1e-5+1e-4*torch.maximum(x.abs(),z.abs())),key
        comparisons[key]=dict(shape=list(x.shape),baseline_repeat_max_abs=float(repeat.max()),hook_max_abs=float(change.max()),bitwise_equal=bool(torch.equal(value,c[key])))
    save_new(RUNTIME/'old_hook_parity_v1.json',dict(status='PASS',source=source,full_original_forward_calls=3,
        token_shapes={str(k):list(v.shape) for k,v in tokens.items()},comparisons=comparisons,
        patch_start_excluded=5,patch_size=14,grid=[37,37],precision='FP16 aggregator, original FP32 heads',
        CUDA_allocator_peak_bytes=torch.cuda.max_memory_allocated(),CUDA_reserved_peak_bytes=torch.cuda.max_memory_reserved(),no_pose_or_F_scan=True))
    print('PASS',comparisons,flush=True)

if __name__=='__main__':main()
