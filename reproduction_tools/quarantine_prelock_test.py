"""Create an auditable quarantine manifest for a legacy pre-lock TEST run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def quarantine(directory: Path) -> Path:
    sentinel = directory / "DO_NOT_USE_FOR_SELECTION"
    if not sentinel.is_file():
        raise RuntimeError(f"quarantine sentinel is missing: {sentinel}")
    manifest = directory / "shard_0.jsonl.manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    original = json.loads(manifest.read_text(encoding="utf-8"))
    output = directory / "prelock_quarantine_manifest.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    payload = {
        "status": "QUARANTINED_PRELOCK_TEST_DIAGNOSTIC",
        "prelock": True,
        "selection_authority": False,
        "official_result_authority": False,
        "official_test_generation_authorized": False,
        "not_for_selection": True,
        "not_for_final_report": True,
        "sentinel": str(sentinel),
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "source_manifest_status": original.get("status"),
        "source_output": original.get("output", str(directory / "shard_0.jsonl")),
        "source_output_sha256": original.get("output_sha256"),
        "reason": "TEST counterfactual was started before FINAL_SELECTION_LOCK",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    output = quarantine(args.directory.resolve())
    print(output)


if __name__ == "__main__":
    main()
