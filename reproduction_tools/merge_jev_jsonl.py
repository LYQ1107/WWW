"""Merge validated JEV JSONL shards without silently overwriting evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from jev_dataset_contract import validate_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def merge(inputs: Sequence[Path], output: Path) -> Mapping[str, object]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    sequences = Counter()
    with output.open("w", encoding="utf-8") as target:
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        validate_record(record, allow_future_gt=True)
                    except Exception as exc:  # noqa: BLE001 - add shard context
                        raise ValueError(f"invalid record in {path}:{line_number}: {exc}") from exc
                    target.write(json.dumps(record, sort_keys=True) + "\n")
                    records += 1
                    sequences[str(record["sequence"])] += 1
    manifest = {
        "status": "PASS",
        "schema_version": 1,
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "source_files": [str(path.resolve()) for path in inputs],
        "source_sha256": {str(path.resolve()): sha256(path) for path in inputs},
        "record_count": records,
        "sequence_count": len(sequences),
        "records_by_sequence": dict(sorted(sequences.items())),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge(args.input, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
