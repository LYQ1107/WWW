"""Verify exact single-worker vs merged deterministic chunk records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev_intra_video_chunking import compare_records_exact


def read_records(path: Path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"record is not an object: {path}")
                records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-records", type=Path, required=True)
    parser.add_argument("--chunk-records", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    single = read_records(args.single_records)
    chunked = []
    for path in args.chunk_records:
        chunked.extend(read_records(path))
    report = compare_records_exact(single, chunked)
    report.update(
        {
            "single_records_path": str(args.single_records.resolve()),
            "chunk_records_paths": [str(path.resolve()) for path in args.chunk_records],
            "canonical_authority": "PASS_ONLY_IF_EXACT_RECORD_MATCHES_AND_SNAPSHOT_PROVENANCE_IS_VALID",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
