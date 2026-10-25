"""Explicit frozen R2 -> V4 research adapter; original strict loader unchanged."""
import json
from pathlib import Path
from .world_object_update_head import MODEL_SCHEMA,WorldObjectUpdateHead
from .world_object_checkpoint import detector_contract
from ..data.worldobject_cache import sha256

R2_SOURCE='6bcdcdb1a6b3683c4e406b0791416a77e5e9e64f165fbad4a61667698b64645f'
FULL_SHA='cfda40050e89b31dd4480ca4d5010e23067a94ba18ea7d0e2425d5935d3f7fb8'
GATE_SHA='363a0a687466bff7a3c8c93b2589c981ceb69a67e36fa8b58953a2d3b41aaf0d'
NO_GEOMETRY_SHA='b08badfc634c9652a34366951c93207fa321deb7d1a2b240a76798f11607053f'
NO_GEOMETRY_GATE_SHA='1796b14b36f9a45381a2ed6067149b746482adceec4b233f9656779ae4c00b1f'
RETAINED_EPOCH1_SHA='d4644ff95cbb7f255f70f155055fba6870863a69e7043f91b920e85ce20693dd'


def load_frozen_r2_head(manifest,raw_lock,device='cpu',*,method='FULL_WORLD_OBJECT_V3'):
    import torch
    path=Path(manifest)
    expected={
        'FULL_WORLD_OBJECT_V3':(FULL_SHA,GATE_SHA),
        'FULL_NO_GEOMETRY_V3':(NO_GEOMETRY_SHA,NO_GEOMETRY_GATE_SHA)}
    if method not in expected:raise ValueError('UNREGISTERED_FROZEN_REFERENCE')
    weight_sha,gate_sha=expected[method]
    if sha256(path)!=gate_sha:raise ValueError('FROZEN_R2_GATE_SHA_MISMATCH')
    gate=json.loads(path.read_text())
    if (gate.get('status')!='FULL_TRAIN_GATE_PASSED' or gate.get('source_sha256')!=R2_SOURCE
            or gate.get('method_id')!=method or gate.get('checkpoint_sha256')!=weight_sha
            or gate.get('final_timestamps')!=28754 or sorted(gate.get('final_scenes',[]))!=list(range(1,25))
            or gate.get('GT_in_model_inputs') is not False
            or gate.get('detector_contract')!=detector_contract(raw_lock)):
        raise ValueError('FROZEN_R2_PROVENANCE_MISMATCH')
    weights=path.parent/gate['checkpoint']
    if sha256(weights)!=weight_sha:raise ValueError('FROZEN_R2_WEIGHT_SHA_MISMATCH')
    payload=torch.load(weights,map_location='cpu',weights_only=True)
    spec=gate['model_config']
    if payload.get('schema')!=MODEL_SCHEMA or payload.get('model_config')!=spec:
        raise ValueError('FROZEN_R2_ARCHITECTURE_MISMATCH')
    head=WorldObjectUpdateHead(spec['visual_dim'],spec['hidden_dim'],spec['geometry_dim'],spec['birth_pair_chunk'])
    head.load_state_dict(payload['state_dict'],strict=True);head.to(device).eval().requires_grad_(False)
    return head,{'checkpoint_sha256':weight_sha,'train_gate_sha256':gate_sha,'method':method,
        'origin_source_sha256':R2_SOURCE,'frozen_parameters':sum(p.numel() for p in head.parameters()),
        'optimizer_used':False,'TRAIN_scenes_previously_seen_by_final_head':True}


def load_retained_epoch1_mechanism_head(r2_root,device='cpu'):
    """Only the separately archived original failure checkpoint, never primary."""
    import torch
    index_path=Path(r2_root)/'audit_snapshots/epoch_archive_index.json'
    records=json.loads(index_path.read_text())['records']
    record=next(x for x in records if x['method']=='FULL_WORLD_OBJECT_V3'
        and x.get('usable_as_selection_epoch_checkpoint') and x['selection_epochs']==1)
    if (record['sha256']!=RETAINED_EPOCH1_SHA or sha256(record['path'])!=RETAINED_EPOCH1_SHA
            or not record.get('resume_metadata_SHA_matched_at_capture')):
        raise ValueError('RETAINED_ORIGINAL_FAILURE_CHECKPOINT_PROOF_REQUIRED')
    payload=torch.load(record['path'],map_location='cpu',weights_only=False)
    head=WorldObjectUpdateHead(1024,128);head.load_state_dict(payload['model'],strict=True)
    head.to(device).eval().requires_grad_(False)
    return head,{'checkpoint_sha256':RETAINED_EPOCH1_SHA,'selection_epoch':1,
        'archive_index_sha256':sha256(index_path),'origin_source_sha256':R2_SOURCE,
        'scope':'exact original t704/t705 TRAIN mechanism only; not primary final-all24 comparison',
        'optimizer_used':False,'TRAIN_scenes_previously_seen_by_final_head':False}
