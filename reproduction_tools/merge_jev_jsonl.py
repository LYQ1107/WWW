"""Merge validated JEV JSONL shards without silently overwriting evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from jev_dataset_contract import validate_record


PROVENANCE_FIELDS = (
    "gmt_checkpoint_sha256",
    "config_sha256",
    "source_commit",
    "counterfactual_engine_version",
    "state_schema_version",
    "utility_definition",
    "association_backend",
    "annotations_sha256",
    "cache_sha256",
    "trace_sha256",
    "horizon",
    "derived_horizons",
    "prelock",
    "selection_authority",
    "official_result_authority",
    "official_test_generation_authorized",
    "official_test_lock_sha256",
    "official_test_selection_protocol_sha256",
)


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
    if not inputs:
        raise ValueError("at least one input shard is required")
    source_manifests = []
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not source_manifest_path.is_file():
            raise FileNotFoundError(source_manifest_path)
        source_manifests.append(
            json.loads(source_manifest_path.read_text(encoding="utf-8"))
        )
    reference = source_manifests[0]
    missing = [field for field in PROVENANCE_FIELDS if field not in reference]
    if missing:
        raise ValueError(
            "REFUSE_TO_MERGE: source shard is missing provenance fields: "
            + ", ".join(missing)
        )
    for index, manifest in enumerate(source_manifests[1:], 1):
        mismatches = {
            field: (reference.get(field), manifest.get(field))
            for field in PROVENANCE_FIELDS
            if reference.get(field) != manifest.get(field)
        }
        if mismatches:
            raise ValueError(
                "REFUSE_TO_MERGE: shard provenance mismatch at index "
                f"{index}: {json.dumps(mismatches, sort_keys=True)}"
            )
    records = 0
    sequences = Counter()
    horizons = set()
    checkpoints = set()
    backends = set()
    official_test_locks = set()
    official_test_selection_protocols = set()
    official_test_authorized = set()
    with output.open("w", encoding="utf-8") as target:
        for path, source_manifest in zip(inputs, source_manifests):
            if source_manifest.get("official_test_lock_sha256"):
                official_test_locks.add(str(source_manifest["official_test_lock_sha256"]))
            if source_manifest.get("official_test_selection_protocol_sha256"):
                official_test_selection_protocols.add(
                    str(source_manifest["official_test_selection_protocol_sha256"])
                )
            if "official_test_generation_authorized" in source_manifest:
                official_test_authorized.add(
                    bool(source_manifest["official_test_generation_authorized"])
                )
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
                    horizons.add(int(record["horizon"]))
                    checkpoints.add(str(record["gmt_checkpoint_sha256"]))
                    state = record.get("state") or {}
                    backends.add(str(state.get("association_backend", "")))
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
        "horizon": next(iter(horizons)) if len(horizons) == 1 else None,
        "horizons": sorted(horizons),
        "gmt_checkpoint_sha256": sorted(checkpoints),
        "association_backends": sorted(backends),
        "official_test_lock_sha256": (
            next(iter(official_test_locks)) if len(official_test_locks) == 1 else None
        ),
        "official_test_selection_protocol_sha256": (
            next(iter(official_test_selection_protocols))
            if len(official_test_selection_protocols) == 1
            else None
        ),
        "official_test_generation_authorized": (
            len(official_test_authorized) == 1
            and next(iter(official_test_authorized)) is True
            and all(
                manifest.get("official_test_generation_authorized") is True
                for manifest in source_manifests
            )
        ),
    }
    for field in PROVENANCE_FIELDS:
        manifest[field] = reference.get(field)
    manifest["provenance_consistency"] = "PASS"
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
