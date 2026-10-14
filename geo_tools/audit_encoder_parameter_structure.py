"""Count exact pinned VGGT module parameters on meta, without another GPU."""
from common import *


def main():
    import torch
    source=RUNTIME/'sources/vggt';sys.path.insert(0,str(source))
    from vggt.models.vggt import VGGT
    original_linspace=torch.linspace
    def cpu_linspace(*args,**kwargs):
        # DINO's constructor reads tiny stochastic-depth rates with .item().
        # Keep only that scalar schedule on CPU; all parameters remain meta.
        kwargs['device']='cpu'
        return original_linspace(*args,**kwargs)
    torch.linspace=cpu_linspace
    try:
        with torch.device('meta'):model=VGGT()
    finally:torch.linspace=original_linspace
    if any(p.device.type!='meta' for p in model.parameters()):raise ValueError('unexpected parameter allocation')
    full=sum(p.numel() for p in model.parameters())
    measured=json.loads((ROOT/'results/feature_screening_manifest.json').read_text())['parameters']
    if full!=measured:raise ValueError('architecture differs from original loaded-model parameter count')
    removed={name:sum(p.numel() for p in getattr(model,name).parameters()) for name in ('point_head','depth_head','track_head')}
    for name in removed:setattr(model,name,None)
    model.eval().requires_grad_(False)
    resident=sum(p.numel() for p in model.parameters())
    if full-sum(removed.values())!=resident:raise ValueError('pruning count mismatch')
    save_json(ROOT/'results/vggt_resident_parameter_structure_v1.json',dict(status='PASS_PARAMETER_STRUCTURE',
        full_official_parameters=full,resident_parameters=resident,removed_parameters=removed,
        trainable_at_inference=sum(p.numel() for p in model.parameters() if p.requires_grad),
        parameter_tensor_bytes_default_fp32=sum(p.numel()*p.element_size() for p in model.parameters()),
        nonparameter_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        module_parameter_devices=sorted({str(p.device) for p in model.parameters()}),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        source_files_sha256={str(source/name):sha256(source/name) for name in ('vggt/models/vggt.py','vggt/models/aggregator.py','vggt/layers/vision_transformer.py')},
        adapter_sha256=sha256(ROOT/'gtr/geometry/vggt_adapter.py'),
        scope='same original VGGT module constructor, then point/depth/track heads removed exactly as FrozenVGGT; full count matches prior actually loaded-model measurement; no checkpoint loading, forward inference, train, or extra GPU',
        limitations='Meta structural parameter and default-dtype payload accounting is not measured CUDA allocation/activation/allocator/process memory. FrozenVGGT uses FP16 autocast on actual V100 while retaining default FP32 weights. Real CUDA peaks remain separate.',provenance=provenance()))
    print(json.dumps(dict(full=full,resident=resident,removed=removed)))


if __name__=='__main__':main()
