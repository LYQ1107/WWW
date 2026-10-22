#!/usr/bin/env python
"""Record machine, source, data, and forbidden-dependency audit facts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=30).stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/environment_audit.json")
    ap.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--dataset", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    ap.add_argument("--panst3r-root", default="/data1/liuyeqiang/PanSt3R_official_20261021")
    ap.add_argument("--must3r-root", default="/data1/liuyeqiang/MUSt3R_official_20261021")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    forbidden = []
    for p in project.rglob("*.py"):
        # Parse imports instead of searching source text.  The audit itself and
        # its unit test necessarily contain the forbidden names as literals;
        # text matching would report those documentation strings as imports.
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"), filename=str(p))
        except SyntaxError:
            continue
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        if any(m == "gtr" or m.startswith("gtr.") or m == "detectron2" or m.startswith("detectron2.") for m in modules):
            forbidden.append(str(p))
    ckpt = Path(args.panst3r_root) / "panst3r_v2_512_5ds.pth"
    out = {
        "python": sys.version,
        "executable": sys.executable,
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"]),
        "running_compute": run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]),
        "disk": run(["df", "-h", "/data1", "/home"]),
        "project": str(project), "dataset": str(Path(args.dataset)),
        "dataset_train_json": (Path(args.dataset) / "annotations/train.json").exists(),
        "panst3r_git": run(["git", "-C", args.panst3r_root, "rev-parse", "HEAD"]),
        "must3r_git": run(["git", "-C", args.must3r_root, "rev-parse", "HEAD"]),
        "panst3r_checkpoint": str(ckpt), "checkpoint_exists": ckpt.exists(),
        "checkpoint_md5": hashlib.md5(ckpt.read_bytes()).hexdigest() if ckpt.exists() else None,
        "forbidden_imports": forbidden,
        "status": "PASS" if not forbidden else "FORBIDDEN_IMPORT_FOUND",
    }
    outpath = Path(args.output); outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if not forbidden else 2


if __name__ == "__main__":
    raise SystemExit(main())
