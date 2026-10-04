#!/usr/bin/env python3
"""Record completion metadata for an isolated proxy inference run."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--source-overlay", type=Path, required=True)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    trace = args.trace if args.trace.is_absolute() else ROOT / args.trace
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    overlay = args.source_overlay if args.source_overlay.is_absolute() else ROOT / args.source_overlay
    result = output / f"inference_{args.dataset}/coco_instances_results.json"
    status = "COMPLETE" if trace.is_file() and trace.stat().st_size > 0 and result.is_file() else "INCOMPLETE"
    payload = {
        "status": status,
        "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "STRUCTURE_SCREENING_ONLY",
        "checkpoint_kind": "PROXY",
        "uses_future_gt": False,
        "dataset": args.dataset,
        "gpu": str(args.gpu),
        "output": str(output),
        "trace": str(trace),
        "trace_bytes": trace.stat().st_size if trace.is_file() else 0,
        "result": str(result),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_overlay": str(overlay),
        "source_overlay_note": "isolated proxy overlay; final numbers require canonical model_20000.pth",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "proxy_inference_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
