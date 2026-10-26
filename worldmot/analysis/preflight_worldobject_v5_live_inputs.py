#!/usr/bin/env python3
"""Read-only readiness for the registered RGB timing; no neural forward or GT."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from PIL import Image

from worldmot.data.visiontrack import VisionTrack
from worldmot.data.worldobject_cache import sha256
from worldmot.models.gmt_detector_adapter import GMT_SOURCE, GMT_PYTHON, GMT_CHECKPOINT_SHA256
from worldmot.models.omega_geometry_backend import OFFICIAL_SOURCE_SHA256, OFFICIAL_OMEGA_CHECKPOINT_SHA256
from worldmot.runtime.world_object_v4_test_contract import install_no_gt_guard
from worldmot.runtime.world_object_v5_contract import require_v5_scope, v5_source_fingerprint, atomic_json


def main():
    parser = argparse.ArgumentParser()
    for name in ("scope-lock", "authorization", "metadata-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep each readiness receipt immutable")
    scope, approval = require_v5_scope(args.scope_lock, args.authorization, "preflight", "train", "B3")
    started, fingerprint = time.monotonic(), v5_source_fingerprint()
    guard = install_no_gt_guard()
    metadata = args.metadata_root / "annotations/train.json"
    data = json.loads(metadata.read_text())
    if data.get("annotations") or len(data["images"]) != 57508:
        raise ValueError("PHYSICAL_ANNOTATION_FREE_FULL_TRAIN_METADATA_REQUIRED")
    dataset = VisionTrack(args.metadata_root, split="train", load_annotations=False)
    lock_path = Path(scope["splits"]["train"]["raw_cache"]) / "DETECTOR_LOCK.json"
    lock = json.loads(lock_path.read_text())
    if sha256(metadata) != lock["metadata_sha256"]:
        raise ValueError("LIVE_IMAGE_METADATA_DIFFERS_FROM_FROZEN_DETECTOR_EXPORT")
    records, chain = [], hashlib.sha256()
    for scene in (4, 8, 12):
        pairs = list(dataset.synchronized_pairs(scene, limit=61))
        if len(pairs) != 61 or [p.frame_id for p in pairs] != list(range(1, 62)):
            raise ValueError("REGISTERED_LIVE_PREFIX_CHANGED")
        for pair in pairs:
            if sorted(pair.images) != [1, 2]:
                raise ValueError("REGISTERED_CAMERA_ORDER_CHANGED")
            for camera, item in sorted(pair.images.items()):
                image_path = dataset.image_path(item)
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    rgb.load()
                    if rgb.size != (item.width, item.height):
                        raise ValueError("RGB_SIZE_DIFFERS_FROM_FROZEN_METADATA")
                record = {"scene": scene, "timestamp": pair.frame_id, "camera": camera,
                          "image_id": item.image_id, "relative_filename": item.file_name,
                          "width": item.width, "height": item.height, "sha256": sha256(image_path)}
                records.append(record)
                chain.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        atomic_json(args.output.with_suffix(".progress.json"), {
            "status": "READ_ONLY_INPUT_PREFLIGHT", "decoded_images": len(records), "last_complete_scene": scene})
        print(json.dumps({"RGB_prefix_ready": scene, "decoded_images": len(records)}), flush=True)
    source_root = Path(GMT_SOURCE)
    common = source_root / "geo_tools/common.py"
    checkpoint = None
    for statement in ast.parse(common.read_text()).body:
        if (isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CHECKPOINT" for t in statement.targets)
                and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "Path"):
            checkpoint = Path(ast.literal_eval(statement.value.args[0]))
    if checkpoint is None:
        raise ValueError("FROZEN_GMT_CHECKPOINT_NOT_AVAILABLE")
    actual_gmt_sha = sha256(checkpoint)
    if actual_gmt_sha != GMT_CHECKPOINT_SHA256 or actual_gmt_sha != lock["checkpoint_sha256"]:
        raise ValueError("FROZEN_GMT_WEIGHT_SHA_CHANGED")
    config_code = r'''
import sys,os,json,hashlib
def guard(event,args):
    if event == 'open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]).replace('\\','/')
        if '/VisionTrack/annotations/' in name or ('/VisionTrack/' in name and ('/gt/' in name or name.endswith('/gt.txt'))):
            raise PermissionError('NO_ORIGINAL_GT_DURING_CONFIG_ONLY_PREFLIGHT')
sys.addaudithook(guard)
root=sys.argv[1]
sys.path[:0]=[root+'/geo_tools',root,root+'/third_party/CenterNet2']
import gtr
from detectron2.config import get_cfg
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
cfg=get_cfg();add_centernet_config(cfg);add_gtr_config(cfg)
cfg.merge_from_file(root+'/configs/VISION_test.yaml')
cfg.MODEL.WEIGHTS=sys.argv[2]
cfg.MODEL.GEO.ENABLED=False
print('V5_CONFIG_ONLY_RESULT='+json.dumps({'config_sha256':hashlib.sha256(cfg.dump().encode()).hexdigest(),'model_instantiated':False,'GT_read':False}))
'''
    child = subprocess.run([GMT_PYTHON, "-c", config_code, str(source_root), str(checkpoint)],
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                           capture_output=True, text=True, timeout=45)
    if child.returncode:
        raise RuntimeError("GMT_CONFIG_DEPENDENCY_PREFLIGHT_FAILED: " + child.stderr[-2000:])
    config = json.loads(next(line.split("=", 1)[1] for line in child.stdout.splitlines()
                             if line.startswith("V5_CONFIG_ONLY_RESULT=")))
    if config["config_sha256"] != lock["config_sha256"]:
        raise ValueError("FROZEN_GMT_CONFIG_CHANGED")
    omega_root = Path("/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega")
    omega_weight = Path("/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt")
    omega_sources = {name: sha256(omega_root / name) for name in OFFICIAL_SOURCE_SHA256}
    if omega_sources != OFFICIAL_SOURCE_SHA256 or sha256(omega_weight) != OFFICIAL_OMEGA_CHECKPOINT_SHA256:
        raise ValueError("VERIFIED_OMEGA512_DEPENDENCY_CHANGED")
    adapter = Path(__file__).resolve().parents[1] / "worldmot/models/gmt_detector_adapter.py"
    if guard["blocked_GT_reads"] or v5_source_fingerprint() != fingerprint:
        raise ValueError("GT_READ_ATTEMPT_OR_FROZEN_SOURCE_CHANGE")
    report = {"status": "REGISTERED_RGB_INPUTS_WEIGHTS_AND_CONFIG_READY_NO_MODEL_FORWARD",
        "UTC": datetime.now(timezone.utc).isoformat(), "approval": approval, "source_sha256": fingerprint,
        "scope_sha256": sha256(args.scope_lock), "metadata_sha256": sha256(metadata),
        "metadata_sha256_equals_frozen_detector_export": True, "preflight_script_sha256": sha256(__file__),
        "TRAIN_scenes": [4, 8, 12], "paired_times_per_scene": 61, "paired_times": 183,
        "decoded_RGB_images": len(records), "image_hash_chain_sha256": chain.hexdigest(),
        "images": records, "warmup_images_are_in_the_same_registered_first5_of_scene4": True,
        "GMT_checkpoint_sha256": actual_gmt_sha, "GMT_config_only_check": config,
        "GMT_adapter_sha256": sha256(adapter), "cached_export_adapter_sha256": lock["adapter_sha256"],
        "adapter_source_hash_equals_cached_export": sha256(adapter) == lock["adapter_sha256"],
        "Omega512_checkpoint_sha256": OFFICIAL_OMEGA_CHECKPOINT_SHA256, "Omega_source_files": omega_sources,
        "actual_model_initialization_or_forward": False, "GT_read": False,
        "LIVE_FPS_measured": False, "tracking_accuracy_measured": False,
        "native_scores_and_all3_final_TRAIN_gates_still_required_before_LIVE": True,
        "readiness_audit_wall_s_not_model_runtime": time.monotonic() - started}
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "decoded_RGB_images": len(records)}), flush=True)


if __name__ == "__main__":
    main()
