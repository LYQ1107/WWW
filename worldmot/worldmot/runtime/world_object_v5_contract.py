"""Explicit V5 authorization and frozen input/configuration provenance."""
import hashlib,json,os
from pathlib import Path
from .experiment_authorization import source_fingerprint,require_experiment_authorization,source_commit
from ..data.worldobject_cache import sha256

BASE="eda0761b15f4049ead1971facd17d846bbab94cf"
SCHEMA="worldmot_v5_registered_scope_v1"
FULL_SHA="cfda40050e89b31dd4480ca4d5010e23067a94ba18ea7d0e2425d5935d3f7fb8"
LEGAL_TEST=[1,2,3,4,6,7,8,9,10,11,15,16,18,19]


def registration_path():return Path(__file__).resolve().parents[2]/"configs/worldobject_v5/registered.json"


def v5_source_fingerprint():
    return hashlib.sha256((source_fingerprint()+"\0"+sha256(registration_path())).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".partial."+str(os.getpid()))
    with temp.open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def require_v5_scope(scope_path,authorization_path,operation,split,variant):
    approved=require_experiment_authorization(authorization_path,operation,split,"V5_"+variant)
    scope=json.loads(Path(scope_path).read_text());auth=json.loads(Path(authorization_path).read_text())
    if (scope.get("schema")!=SCHEMA or scope.get("base_commit")!=BASE
        or scope.get("v5_source_sha256")!=v5_source_fingerprint()
        or scope.get("registered_config_sha256")!=sha256(registration_path())
        or auth.get("scope_lock_sha256")!=sha256(scope_path)
        or variant not in scope.get("variants",[]) or split not in scope.get("splits",{})):
        raise PermissionError("V5_REGISTERED_SOURCE_OR_INPUT_SCOPE_MISMATCH")
    if operation=="train" and (split!="train" or variant not in {"B2","B3","B4"}):raise PermissionError("V5_MEMBERSHIP_TRAIN_SCOPE_ONLY")
    if scope.get("TEST_parameter_selection") is not False:raise PermissionError("TEST_SELECTION_FORBIDDEN")
    return scope,approved


def verify_scene_scope(scope,split,scene,cache):
    entry=scope["splits"][split]["scenes"][str(scene)]
    if (cache.manifest.get("status")!="COMPLETE" or sha256(cache.path/"manifest.json")!=entry["support_manifest_sha256"]
            or cache.manifest["timestamps"]!=entry["timestamps"]):raise ValueError("V5_FROZEN_SCENE_SCOPE_MISMATCH")
    raw_path=Path(scope["splits"][split]["raw_cache"])/f"scene_{scene:02d}/manifest.json"
    if sha256(raw_path)!=entry["raw_manifest_sha256"]:raise ValueError("V5_FROZEN_RAW_MANIFEST_CHANGED")
    return entry


def state_tensor_sha256(model):
    h=hashlib.sha256()
    for key,tensor in sorted(model.state_dict().items()):
        h.update(key.encode()+b"\0");h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initial_membership_model(config,seed,device):
    import random,numpy as np,torch
    from ..models.world_object_membership_head import WorldObjectMembershipHead
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if str(device).startswith("cuda"):torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    model=WorldObjectMembershipHead(config.feature_dim,config.hidden_dim,config.geometry_dim,config.pair_chunk,config.feature_dropout).to(device)
    return model


def load_v5_membership_gate(gate_path,config,device="cpu",expected_sha=None):
    import torch
    from ..models.world_object_membership_head import MODEL_SCHEMA,WorldObjectMembershipHead
    path=Path(gate_path);gate=json.loads(path.read_text())
    if (gate.get("status")!="FULL_TRAIN_GATE_PASSED" or gate.get("variant")!=config.variant
            or gate.get("v5_source_sha256")!=v5_source_fingerprint() or gate.get("tracking_config")!=config.to_dict()
            or gate.get("final_scenes")!=list(range(1,25)) or gate.get("final_timestamps")!=28754
            or gate.get("history_mode")!="predicted_from_first_timestamp" or gate.get("GT_in_model_inputs") is not False
            or gate.get("final_epochs") not in {12,16} or gate.get("prediction_heads")!=["membership","per_camera_no_observation"]):
        raise ValueError("V5_FULL_TRAIN_GATE_REQUIRED")
    coverage=gate.get("final_epoch_coverage",[])
    if (len(coverage)!=gate["final_epochs"] or gate.get("final_total_training_timestamps")!=28754*gate["final_epochs"]
        or any(sorted(x["scene"] for x in epoch)!=list(range(1,25)) or sum(x["timestamps"] for x in epoch)!=28754 for epoch in coverage)):
        raise ValueError("V5_EVERY_FINAL_EPOCH_COVERAGE_REQUIRED")
    spec=gate["model_config"]
    expected={"visual_dim":config.feature_dim,"hidden_dim":config.hidden_dim,"geometry_dim":config.geometry_dim,
              "pair_chunk":config.pair_chunk,"dropout":config.feature_dropout}
    if any(spec.get(k)!=v for k,v in expected.items()):raise ValueError("V5_REGISTERED_NETWORK_CAPACITY_MISMATCH")
    weights=path.parent/gate["checkpoint"];digest=sha256(weights)
    if digest!=gate["checkpoint_sha256"] or (expected_sha is not None and digest!=expected_sha):raise ValueError("V5_FROZEN_WEIGHT_SHA_MISMATCH")
    payload=torch.load(weights,map_location="cpu",weights_only=True)
    if payload.get("schema")!=MODEL_SCHEMA or payload.get("model_config")!=gate["model_config"]:raise ValueError("V5_ARCHITECTURE_MISMATCH")
    model=WorldObjectMembershipHead(spec["visual_dim"],spec["hidden_dim"],spec["geometry_dim"],spec["pair_chunk"],spec["dropout"])
    model.load_state_dict(payload["state_dict"],strict=True);model.to(device).eval().requires_grad_(False)
    return model,{"checkpoint_sha256":digest,"train_gate_sha256":sha256(path),"final_epochs":gate["final_epochs"],"optimizer_used":False}
