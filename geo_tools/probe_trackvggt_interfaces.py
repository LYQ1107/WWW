"""Real two-view frozen-weight interface probe; no GT or new model training."""
import argparse,importlib.util,time
import numpy as np
import torch
from common import *

OLD=Path('/data1/liuyeqiang/WWW_vggt_geo_runtime')
NEW=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['official','det'],required=True);p.add_argument('--tag',required=True)
    a=p.parse_args();out=NEW/'interfaces'/a.tag
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source=OLD/'sources'/('vggt' if a.backend=='official' else 'VGGT-Det')
    sys.path.insert(0,str(source))
    from vggt.utils.load_fn import load_and_preprocess_images
    from gtr.geometry.trackvggt_features import FrozenTrackVGGT,explicit_layer_map,LAYERS
    seed_all(20261014);data=train_data();records=video_records(data,5,max_frames=1)
    images=load_and_preprocess_images([row['file_name'] for row in records],mode='pad')[None]
    if images.shape[1]!=2:raise ValueError('probe expects physical two-view group')
    checkpoint=OLD/'weights/vggt_1b.pt';started=time.perf_counter();det_check=None
    if a.backend=='official':
        encoder=FrozenTrackVGGT(checkpoint);encoder.capture_input=True
        layers,attention,meta=encoder.extract(images)
        # Compare our chunked statistic to pinned VGGT-Det's actual Attention
        # implementation at exactly the same real input and learned Q/K/RoPE.
        path=OLD/'sources/VGGT-Det/vggt/layers/attention.py'
        spec=importlib.util.spec_from_file_location('trackvggt_det_attention_reference',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        ref=module.Attention(1024,num_heads=16,qk_norm=True,rope=encoder.model.rope).cuda().eval()
        current=encoder.model.frame_blocks[0].attn;ref.load_state_dict(current.state_dict(),strict=True)
        x,pos=encoder.last_input
        with torch.inference_mode(),torch.autocast('cuda',dtype=encoder.dtype):
            original=current(x,pos=pos);expected=ref(x,pos=pos,save_vis_attn=True)
        native=ref.last_attn.float().mean((1,2))[:,meta['patch_start_idx']:][None]
        det_check=dict(actual_attention_output_exact=bool(torch.equal(original,expected)),
            actual_attention_output_max_abs=float((original.float()-expected.float()).abs().max()),
            actual_attention_output_allclose=bool(torch.allclose(original.float(),expected.float(),rtol=.002,atol=.0001)),
            incoming_attention_max_abs=float((native-attention).abs().max()),
            incoming_attention_allclose=bool(torch.allclose(native,attention,rtol=.003,atol=1e-6)),
            reference_source=str(path),reference_sha256=sha256(path))
        print(json.dumps(dict(attention_diagnostic=det_check)),flush=True)
        assert det_check['actual_attention_output_exact'] and det_check['incoming_attention_allclose']
        frozen=all(not v.requires_grad for v in encoder.model.parameters());params=sum(v.numel() for v in encoder.model.parameters())
    else:
        from vggt.models.aggregator import Aggregator
        model=Aggregator();full=torch.load(checkpoint,map_location='cpu');state={k.removeprefix('aggregator.'):v for k,v in full.items() if k.startswith('aggregator.')}
        strict=model.load_state_dict(state,strict=True);del state,full
        model=model.cuda().eval().requires_grad_(False)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            values,start,native_attention=model(images.cuda(),if_norm=True,if_detach=True,if_only_last_layer=False,if_use_atten_sample=True,if_task_query=True)
        layers=explicit_layer_map(values,start,images.shape[-2:],compact_layers=LAYERS)
        attention=model.frame_blocks[0].attn.last_attn.float().mean((1,2))[:,start:][None]
        assert torch.allclose(attention[0],native_attention.float(),rtol=.003,atol=1e-6)
        meta=dict(patch_start_idx=start,layer_ids=list(LAYERS),image_hw=list(images.shape[-2:]))
        frozen=all(not v.requires_grad for v in model.parameters());params=sum(v.numel() for v in model.parameters())
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    # Retain only32 deterministic probe positions per view, never dense maps or
    # a full attention matrix. This is interface validation, not AG selection.
    indices=torch.linspace(0,layers[4].shape[-2]-1,32).long().to(layers[4].device)
    np.savez_compressed(out/'probe_positions.npz',**{str(k):v[0,:,indices].float().cpu().numpy() for k,v in layers.items()},attention=attention.cpu().numpy())
    report=dict(status='PASS_REAL_TWO_VIEW_INTERFACE',backend=a.backend,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        source_sha256={str(path):sha256(path) for path in [source/'vggt/models/aggregator.py',source/'vggt/layers/attention.py',source/'vggt/layers/block.py']},
        checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),input_shape=list(images.shape),input_range=[float(images.min()),float(images.max())],
        image_ids=[row['image_id'] for row in records],input_paths=[row['file_name'] for row in records],
        layer_shapes={str(k):list(v.shape) for k,v in layers.items()},attention_shape=list(attention.shape),metadata=meta,
        strict_aggregator_weights_loaded=True,frozen=frozen,parameters=params,det_attention_equivalence=det_check,
        elapsed_including_model_load_seconds=seconds,cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
        scope='real first TRAIN5 two-view group, backbone-only strict compatible official weights;32 positions retained per view, not full-video accuracy/runtime or geometry/identity efficacy',provenance=provenance())
    report['probe_features_sha256']=sha256(out/'probe_positions.npz');save_json(out/'manifest.json',report)
    print(json.dumps({k:report[k] for k in ['status','backend','input_shape','layer_shapes','attention_shape','det_attention_equivalence','elapsed_including_model_load_seconds']}),flush=True)


if __name__=='__main__':main()
