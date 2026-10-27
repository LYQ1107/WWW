"""Generated-input FP32 checks for the registered final V100 model path.

No dataset, trained checkpoint, optimizer or perception model is used. The
selection workers keep running, and no generated weights are saved.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from worldmot.data.worldobject_cache import sha256
from worldmot.runtime.world_object_v4_test_contract import install_no_gt_guard
from worldmot.runtime.world_object_v5_contract import (
    atomic_json, initial_membership_model, registration_path, state_tensor_sha256,
    v5_source_fingerprint)
from worldmot.tracking.world_object_v5_config import registered_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve any previous hardware check")
    guard = install_no_gt_guard()
    before = v5_source_fingerprint()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("USE_THE_EXCLUSIVELY_RESERVED_GPU_UUID")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    for line in apps.splitlines():
        uuid, pid = [x.strip() for x in line.split(",")]
        if uuid == args.gpu_uuid and int(pid) != os.getpid():
            raise RuntimeError("GPU_BECAME_OCCUPIED_DO_NOT_INTERFERE_WITH_OTHER_USERS")
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SINGLE_RESERVED_CUDA_DEVICE_REQUIRED")
    name = torch.cuda.get_device_name(0)
    if "V100" not in name:
        raise ValueError("FINAL_DEVICE_FAMILY_MUST_BE_V100")
    reg = json.loads(registration_path().read_text())
    cal = json.loads(args.calibration.read_text())
    configs = {v: registered_config(v, cal) for v in ("B2", "B3", "B4")}
    cpu_sha, architectures, counts = {}, {}, {}
    for variant, config in configs.items():
        model = initial_membership_model(config, reg["seed"], "cpu")
        cpu_sha[variant] = state_tensor_sha256(model)
        architectures[variant] = model.configuration()
        counts[variant] = sum(p.numel() for p in model.parameters())
        del model
    if len(set(cpu_sha.values())) != 1 or set(counts.values()) != {606468}:
        raise ValueError("REGISTERED_VARIANTS_MUST_SHARE_INITIALIZATION_AND_CAPACITY")
    expected_initial = "3fb3f3d6336e8aa32d635ecff3c05929b0d4b8f265f4a0ccf56771814d13ed23"
    if set(cpu_sha.values()) != {expected_initial}:
        raise ValueError("REGISTERED_INITIAL_TENSORS_CHANGED")
    cases = []
    for device in ("cpu", "cuda:0"):
        model = initial_membership_model(configs["B3"], reg["seed"], device).train()
        if state_tensor_sha256(model) != expected_initial:
            raise ValueError("CPU_AND_CUDA_INITIAL_TENSORS_DIFFER")
        for m, n in ((0, 8), (8, 0), (96, 72), (192, 144)):
            generator = torch.Generator(device="cpu").manual_seed(reg["seed"] + 17 + m + n)
            random = lambda *shape: torch.randn(shape, generator=generator, dtype=torch.float32).to(device)
            inputs = (F.normalize(random(m, 2, 1024), dim=-1), F.normalize(random(n, 1024), dim=-1),
                      random(m, n, 32), random(n, 6), random(m, 8),
                      torch.arange(n, device=device, dtype=torch.long) % 2)
            model.zero_grad(set_to_none=True)
            outputs = model(*inputs)
            if outputs["membership_logits"].shape != (m, n) or outputs["no_observation_logits"].shape != (m, 2):
                raise ValueError("REGISTERED_MEMBERSHIP_AND_PER_CAMERA_NULL_SHAPES_REQUIRED")
            if any(x.dtype != torch.float32 or not bool(torch.isfinite(x).all()) for x in outputs.values()):
                raise ValueError("FP32_FINITE_OUTPUTS_REQUIRED")
            loss = sum(x.square().sum() / max(1, x.numel()) for x in outputs.values())
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(g.dtype != torch.float32 or not bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError("FP32_FINITE_BACKWARD_REQUIRED")
            cases.append({"device": device, "history_objects": m, "observations": n,
                          "membership_shape": [m, n], "camera_NULL_shape": [m, 2],
                          "finite_outputs": True, "finite_gradients": True, "dtype": "float32"})
        if state_tensor_sha256(model) != expected_initial:
            raise ValueError("A_BACKWARD_ONLY_CHECK_MUST_NOT_CHANGE_WEIGHTS")
        del model
    torch.cuda.synchronize()
    if guard["blocked_GT_reads"] or v5_source_fingerprint() != before:
        raise ValueError("ORIGINAL_GT_READ_ATTEMPT_OR_FROZEN_SOURCE_CHANGED")
    proof = {"status": "PASSED_GENERATED_FP32_CPU_AND_V100_FORWARD_BACKWARD",
        "source_sha256": sha256(__file__), "v5_source_sha256": before,
        "registered_config_sha256": sha256(registration_path()), "calibration_sha256": sha256(args.calibration),
        "GPU_UUID": args.gpu_uuid, "GPU_name": name, "torch_version": torch.__version__,
        "CUDA_version": torch.version.cuda, "CUDA_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "initial_tensor_sha256_per_variant": cpu_sha, "parameters_per_variant": counts,
        "architecture_per_variant": architectures, "cases": cases,
        "original_images_annotations_perception_caches_or_trained_checkpoints_read": False,
        "optimizer_created": False, "weights_updated_or_saved": False,
        "selection_training_restarted_or_modified": False, "GT_read_attempts": guard["blocked_GT_reads"],
        "real_training_throughput_or_RGB_FPS_measured": False,
        "membership_head_hardware_readiness_only": True}
    atomic_json(args.output, proof)
    print(json.dumps({"status": proof["status"], "cases": len(cases), "GPU": name,
                      "peak_allocated_bytes": proof["CUDA_peak_allocated_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
