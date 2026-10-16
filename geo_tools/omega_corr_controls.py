"""Frozen old-VGGT and independent DINO with explicit native pad/14 coordinate maps."""
from pathlib import Path
import subprocess
import sys
import time
import torch
from omega_corr_common import sha256
from omega_modules import geometry_module

OLD_SOURCE=Path('/data1/liuyeqiang/WWW_vggt_geo_runtime/sources/vggt')
DINO_SOURCE=Path('/data1/liuyeqiang/WWW_vggt_geo_runtime/sources/dinov2')
WEIGHTS=Path('/data1/liuyeqiang/WWW_vggt_geo_runtime/weights')


class FrozenControl:
    def __init__(self,backend):
        self.backend=backend;self.calls_attempted=0;self.calls_completed=0
        expected_source='a288dd0f14786c93483e45524328726ab7b1b4ce' if backend=='old_vggt' else '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
        source=OLD_SOURCE if backend=='old_vggt' else DINO_SOURCE
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==expected_source
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
        sys.path.insert(0,str(OLD_SOURCE));sys.path.insert(0,str(source))
        if backend=='old_vggt':
            from vggt.models.vggt import VGGT
            path=WEIGHTS/'vggt_1b.pt';expected='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
            constructor=VGGT
        elif backend=='dino':
            from dinov2.hub.backbones import dinov2_vitl14_reg
            path=WEIGHTS/'dinov2_vitl14_reg4.pth';expected='36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51'
            constructor=lambda:dinov2_vitl14_reg(pretrained=False)
        else:raise ValueError(backend)
        assert sha256(path)==expected
        model=constructor().eval().requires_grad_(False)
        state=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
        self.state_keys=len(state);del state
        self.model=model.cuda();self.source_commit=expected_source;self.weight_sha256=expected
        self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

    def prepare(self,paths,sizes):
        from vggt.utils.load_fn import load_and_preprocess_images
        images=load_and_preprocess_images(paths,mode='pad')
        mapping=geometry_module('omega_adapter').PixelMapping
        pad=geometry_module('roi_feature_extractor').PadMapping
        maps=[]
        for w,h in sizes:
            rw,rh=pad(w,h).resized
            maps.append(mapping(w,h,0,0,w,h,rw,rh,(518-rw)//2,(518-rh)//2,518,518))
        return images,maps

    @torch.inference_mode()
    def encode(self,images,layers):
        tokens={};handle=None
        if self.backend=='old_vggt':
            def capture(module,args,output):
                values,start=output;assert start==5
                for layer in layers:
                    v=values[layer][:,:,start:];assert tuple(v.shape[0:3])==(1,2,37*37)
                    assert v.shape[-1]==2048 and torch.isfinite(v).all();tokens[layer]=v.detach()
            handle=self.model.aggregator.register_forward_hook(capture)
        torch.cuda.synchronize();started=time.perf_counter();self.calls_attempted+=1
        try:
            with torch.autocast('cuda',dtype=torch.float16):
                if self.backend=='old_vggt':outputs=self.model(images.cuda())
                else:outputs=self.model.forward_features((images.cuda()-self.mean)/self.std)
            if self.backend=='dino':
                v=outputs['x_norm_patchtokens'];assert tuple(v.shape)==(2,37*37,1024)
                tokens[23]=v[None].detach()
            for v in outputs.values():
                if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
            torch.cuda.synchronize();self.calls_completed+=1
        finally:
            if handle is not None:handle.remove()
        return tokens,time.perf_counter()-started
