"""Strict V3 full-TRAIN checkpoint gate. Legacy W4/W5 weights are incompatible."""
from __future__ import annotations
import json
from pathlib import Path
from ..data.worldobject_cache import sha256
from .world_object_update_head import MODEL_SCHEMA, WorldObjectUpdateHead

GATE_SCHEMA="worldobject_identity_joint_birth_fulltrain_gate_v3_1"


def load_identity_checkpoint(manifest_path, config, device="cpu"):
    import torch
    path=Path(manifest_path)
    report=json.loads(path.read_text())
    if (report.get("schema")!=GATE_SCHEMA or report.get("status")!="FULL_TRAIN_GATE_PASSED"
            or report.get("model_config",{}).get("schema")!=MODEL_SCHEMA
            or report.get("method_id")!=config.checkpoint_method_id
            or report.get("feature_dim")!=config.feature_dim
            or sorted(report.get("final_scenes",[]))!=list(range(1,25))
            or report.get("final_timestamps")!=28754
            or not isinstance(report.get("seed"),int) or not report.get("training_budget")
            or report.get("identity_supervision_observations",0)<=0
            or not report.get("source_commit") or not report.get("detector_contract")
            or not report.get("scope_lock_sha256")
            or report.get("tracking_config")!={**config.to_dict(),"method_id":config.checkpoint_method_id}
            or report.get("GT_in_model_inputs") is not False):
        raise ValueError("INCOMPATIBLE_OR_INCOMPLETE_V3_TRAIN_GATE")
    from ..runtime.experiment_authorization import source_fingerprint
    if report.get("source_sha256")!=source_fingerprint():
        raise ValueError("CHECKPOINT_SOURCE_SHA_MISMATCH")
    weights=path.parent/report["checkpoint"]
    if sha256(weights)!=report["checkpoint_sha256"]:
        raise ValueError("CHECKPOINT_SHA_MISMATCH")
    spec=report["model_config"]
    model=WorldObjectUpdateHead(spec["visual_dim"],spec["hidden_dim"],spec["geometry_dim"],spec["birth_pair_chunk"])
    payload=torch.load(weights,map_location="cpu",weights_only=True)
    if payload.get("schema")!=MODEL_SCHEMA or payload.get("model_config")!=spec:
        raise ValueError("LEGACY_OR_MISMATCHED_CHECKPOINT_PAYLOAD")
    model.load_state_dict(payload["state_dict"],strict=True)
    model.to(device).eval()
    return model,report


def detector_contract(lock):
    keys=("checkpoint_sha256","config_sha256","adapter_sha256","roi_feature_dim","threshold","precision")
    if any(k not in lock for k in keys):
        raise ValueError("INCOMPLETE_FROZEN_DETECTOR_CONTRACT")
    return {k:lock[k] for k in keys}
