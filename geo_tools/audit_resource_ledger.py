"""Recorded model sizes and actual process/cache measurements, with scope."""
import csv
from common import *


def main():
    import torch,gtr
    import layers
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from centernet.config import add_centernet_config
    from gtr.config import add_gtr_config
    cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg)
    cfg.merge_from_file(str(ROOT/'configs/VISION_test.yaml'))
    cfg.MODEL.DEVICE='cpu'
    # The inherited training-only loss constructor unconditionally requests
    # CUDA even for CPU model construction. Its centers are closure tensors,
    # not registered inference parameters. Keep identical loss shapes on CPU
    # for this read-only count; never patch production inference or weights.
    original_center=layers.CenterLoss
    layers.CenterLoss=lambda num_classes,feat_dim,use_gpu:original_center(num_classes,feat_dim,use_gpu=False)
    model=build_model(cfg)
    layers.CenterLoss=original_center
    state=torch.load(CHECKPOINT,map_location='cpu')['model']
    model.load_state_dict(state,strict=True)
    model.eval().requires_grad_(False)
    parameters=dict(gmt=dict(total=sum(p.numel() for p in model.parameters()),
                             trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             strict_checkpoint_keys=len(state),scope='actual complete CPU model, fixed checkpoint strict load, frozen; inherited training-only unregistered center-loss tensors constructed on CPU instead of forced CUDA'))
    del model,state
    sources={}
    for key,name in [('vggt','feature_screening_manifest.json'),('dinov2','dinov2_screening_manifest.json')]:
        path=ROOT/'results'/name;report=json.loads(path.read_text());sources[str(path)]=sha256(path)
        parameters[key]=dict(total=report['parameters'],trainable=0,
            scope=('complete official model before removing unused depth/point/track heads; resident pruned parameter count not measured by this artifact' if key=='vggt' else 'recorded actual independent frozen DINO model'))
    resident_path=ROOT/'results/vggt_resident_parameter_structure_v1.json'
    if resident_path.exists():
        structure=json.loads(resident_path.read_text());sources[str(resident_path)]=sha256(resident_path)
        if structure['status']!='PASS_PARAMETER_STRUCTURE' or structure['full_official_parameters']!=parameters['vggt']['total']:
            raise ValueError('incompatible resident parameter structure')
        parameters['vggt'].update(resident_parameters=structure['resident_parameters'],
            resident_parameter_tensor_bytes_default_fp32=structure['parameter_tensor_bytes_default_fp32'],
            scope='total is original official pre-pruning count; resident_parameters is exact pinned module count after removing point/depth/track heads, structurally verified on meta against prior full loaded-model count; storage estimate is separate from measured CUDA peaks')
    head_path=ROOT/'results/low_rank_three_backbones_train_v1_manifest.json'
    heads=json.loads(head_path.read_text())['results'];sources[str(head_path)]=sha256(head_path)
    parameters['heads']=[dict(backbone=row['backbone'],seed=row['seed'],parameters=row['parameters'],
                              output_dimensions=row['settings']['output_dim']*(2 if row['settings']['symmetric'] else 1),
                              checkpoint_sha256=row['checkpoint_sha256'],trainable_at_inference=0) for row in heads]
    rows=[]
    for family in ('online','retrieval'):
        for path in sorted((RUNTIME/family).glob('*/manifest.json')):
            run=json.loads(path.read_text())
            if not run.get('full_video'):continue
            if family=='online' and sha256(path.parent/'predictions.json')!=run['predictions_sha256']:
                raise ValueError('changed online prediction: '+str(path))
            cache=run.get('target_feature_cache') or {}
            index=run.get('index') or {}
            history=run.get('retained_tracker_payload_bytes')
            row=dict(family=family,tag=path.parent.name,source_manifest=str(path),manifest_sha256=sha256(path),
                     commit=run['commit'],video_ids=run.get('video_ids',[run.get('video_id')]),
                     speed_eligible=run.get('speed_eligible',True),images=run['images'],camera_image_fps=run['camera_image_fps'],
                     peak_cuda_allocated_bytes=run.get('peak_gpu_bytes'),peak_cuda_reserved_bytes=run.get('peak_reserved_gpu_bytes'),
                     peak_host_rss_bytes=run.get('peak_host_rss_bytes'),
                     target_cache_peak_payload_bytes=cache.get('peak_payload_bytes'),
                     index_peak_payload_bytes=index.get('peak_payload_bytes'),
                     index_payload_cap_bytes=index.get('max_payload_bytes'),
                     final_original_tracker_retained_payload_bytes=sum(s['total_bytes'] for s in history.values()) if history is not None else None)
            rows.append(row)
    output=ROOT/'results/resource_ledger_v1.json'
    save_json(output,dict(status='MEASURED_SCOPE_LEDGER_NOT_FINAL_MATRIX',parameters=parameters,runs=rows,parameter_sources_sha256=sources,
        scopes=dict(cuda='PyTorch per-process allocator peak, not all processes or complete device occupancy',
                    host='process lifetime maximum RSS where recorded, including initialization; not synchronized with CUDA/cache peak',
                    original_tracker='deduplicated retained tensor/NumPy storage at final publication, not lifetime peak and not Python object overhead; original GMT history is unchanged and unbounded',
                    cache='target vector cache and retrieval index are separate payloads, not total system memory caps',
                    missing='null means not measured; never substitute zero; do not add noncoincident peaks'),provenance=provenance()))
    with (ROOT/'results/resource_ledger_v1.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(gmt=parameters['gmt'],complete_runs=len(rows),output=str(output))))


if __name__=='__main__':main()
