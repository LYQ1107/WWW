#!/usr/bin/env python3
"""Snapshot selected complete video IDs from a growing proxy trace.

The snapshot is explicitly partial screening data.  It is useful for bringing
up the offline Oracle/policy code while the full frozen inference continues;
it must never be used as a final benchmark result.
"""

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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-id", type=int, action="append", required=True)
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    selected = set(args.video_id)
    if args.max_events is not None and args.max_events < 1:
        raise ValueError("--max-events must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped_partial_lines = 0
    with source.open(encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                skipped_partial_lines += 1
                continue
            video_id = int(event.get("context", {}).get("video_id", -1))
            if video_id in selected:
                writer.write(json.dumps(event, sort_keys=True) + "\n")
                count += 1
                if args.max_events is not None and count >= args.max_events:
                    break
    manifest = {
        "status": "PARTIAL_PROXY",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "source_sha256_at_snapshot": sha256(source),
        "output": str(output.resolve()),
        "video_ids": sorted(selected),
        "events": count,
        "max_events": args.max_events,
        "skipped_partial_lines": skipped_partial_lines,
        "uses_future_gt": False,
        "scope": "EARLY_SCREENING_ONLY",
        "final_numbers_allowed": False,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
