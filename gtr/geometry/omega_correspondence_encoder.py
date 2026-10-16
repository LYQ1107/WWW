"""Read-only hook on the verified official full model, never changes its outputs."""
import time
import torch
from .omega_adapter import OmegaAdapter,preprocess_pair


class OmegaCorrespondenceEncoder(OmegaAdapter):
    def __init__(self,*args,layers=(11,17,23),**kwargs):
        super().__init__(*args,**kwargs)
        self.layers=tuple(layers)
        if not set(self.layers)<={4,11,17,23}:raise ValueError('only official cached layers permitted')
        self.captured=None

    @torch.inference_mode()
    def encode(self,images,camera_ids,timestamp,kind='FULL_IMAGE_MULTIVIEW'):
        if images.ndim==4:images=images.unsqueeze(0)
        if images.ndim!=5 or images.shape[0]!=1 or len(camera_ids)!=images.shape[1]:raise ValueError('explicit view axis required')
        if kind=='FULL_IMAGE_MULTIVIEW' and len(set(camera_ids))!=len(camera_ids):raise ValueError('duplicate physical camera requires artificial-control label')
        saved={}
        def capture(module,args,output):
            values,start=output
            assert start==17==module.patch_token_start and len(values)==24
            assert {i for i,v in enumerate(values) if v is not None}=={4,11,17,23}
            H,W=args[0].shape[-2:];count=(H//16)*(W//16)
            for layer in self.layers:
                v=values[layer]
                assert tuple(v.shape)==(1,len(camera_ids),count+17,2048)
                patch=v[:,:,start:]
                if not torch.isfinite(patch).all():raise ValueError('nonfinite actual patch tokens')
                saved[layer]=patch.detach()
        handle=self.model.aggregator.register_forward_hook(capture)
        started=time.perf_counter();torch.cuda.synchronize(self.device)
        try:
            self.calls_attempted+=1
            with torch.autocast('cuda',dtype=self.amp_dtype):predictions=self.model(images.to(self.device))
            torch.cuda.synchronize(self.device);self.calls_completed+=1
        finally:handle.remove()
        for v in predictions.values():
            if not torch.isfinite(v).all():raise ValueError('nonfinite original model output')
        return dict(tokens=saved,predictions=predictions,camera_ids=list(camera_ids),timestamp=timestamp,
            kind=kind,patch_grid=(images.shape[-2]//16,images.shape[-1]//16),patch_token_start=17,
            CUDA_forward_and_hook_seconds=time.perf_counter()-started,precision=str(self.amp_dtype),
            dimensions=dict(frame_half=1024,interframe_half=1024,concat=2048))

    def prepare(self,paths):return preprocess_pair(self.source,paths,512)
