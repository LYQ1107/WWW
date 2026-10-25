#!/usr/bin/env python3
"""Freeze source, data and resource provenance for the new experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--repo", required=True); p.add_argument("--data-audit", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); repo = Path(a.repo).resolve()
    files = ["docs/GMT_POINT_REPRODUCTION_20261024.md", "worldmot/worldmot/models/omega_geometry_backend.py",
             "worldmot/worldmot/models/gmt_detection_cache.py", "worldmot/worldmot/runtime/online_worldmot.py",
             "worldmot/scripts/cache_worldobject_support.py", "worldmot/scripts/orchestrate_worldobject_six_gpu.py"]
    files += [str(x.relative_to(repo)) for x in repo.glob("worldmot/worldmot/models/world_object_support.py")]
    files += [str(x.relative_to(repo)) for x in repo.glob("worldmot/worldmot/models/multi_view_world_object.py")]
    files += [str(x.relative_to(repo)) for x in repo.glob("worldmot/worldmot/tracking/persistent_world_object_bank.py")]
    hashes = {f: sha(repo / f) for f in files if (repo / f).is_file()}
    def cmd(c):
        try: return subprocess.check_output(c, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc: return f"ERROR:{type(exc).__name__}:{exc}"
    result = {"schema": "worldobject_scope_lock_v1", "created_at": time.time(), "repo": str(repo),
              "git_head": cmd(["git", "-C", str(repo), "rev-parse", "HEAD"]),
              # The host Git is old enough that `branch --show-current` is
              # unavailable; rev-parse is supported by both old and new Git.
              "git_branch": cmd(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"]),
              "source_hashes": hashes, "data_audit": str(Path(a.data_audit).resolve()),
              "data_audit_sha256": sha(Path(a.data_audit)),
              "old_queue_policy": "STOP_NEW_MULTI_SEED; no old V2 queue restart",
              "gt_policy": "raw_GT_immutable; official_metrics_only_legal_original_GT_scenes",
              "runtime": {"nvidia_smi": cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv"]),
                          "compute_apps": cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"]),
                          "df": cmd(["df", "-h", "/data1/liuyeqiang/nas/A100"])}}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True)); os.replace(tmp, out)
    print(json.dumps({"output": str(out), "git_head": result["git_head"], "files": len(hashes)}, indent=2))


if __name__ == "__main__": main()

