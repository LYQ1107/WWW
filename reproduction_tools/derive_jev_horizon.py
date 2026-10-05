"""Derive policy-horizon JSONL datasets from one max-horizon rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jev_dataset_tools import make_record
from jev_dataset_contract import validate_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def derive(source: Path, output: Path, horizon: int) -> Mapping[str, Any]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    source_manifest_path = Path(str(source) + ".manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "PASS":
        raise ValueError("source max-horizon manifest is not PASS")
    if int(horizon) not in {int(value) for value in source_manifest.get("derived_horizons", ())}:
        raise ValueError("source rollout does not contain the requested cumulative horizon")

    records = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as source_handle, output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source_handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record, allow_future_gt=True)
            horizon_outcomes = record.get("horizon_outcomes") or {}
            outcomes = horizon_outcomes.get(str(horizon))
            if not isinstance(outcomes, Mapping):
                raise ValueError(f"missing horizon outcomes at {source}:{line_number}")
            labeling = record.get("labeling") or {}
            derived = make_record(
                dataset=record["dataset"],
                sequence=record["sequence"],
                frame=int(record["frame"]),
                view=int(record["view"]),
                question_type=record["question_type"],
                state=record["state"],
                legal_actions=record["legal_actions"],
                action_outcomes=outcomes,
                gmt_checkpoint_sha256=record["gmt_checkpoint_sha256"],
                horizon=int(horizon),
                temperature=float(labeling.get("temperature", 1.0)),
                tie_tolerance=float(labeling.get("tie_tolerance", 1e-8)),
            )
            # Keep the max-rollout evidence available for audit; only the
            # selected action_outcomes/horizon fields change in this view.
            derived["horizon_outcomes"] = horizon_outcomes
            derived["derived_from_max_horizon"] = int(source_manifest["horizon"])
            validate_record(derived, allow_future_gt=True)
            target.write(json.dumps(derived, sort_keys=True) + "\n")
            records += 1

    manifest = {
        "status": "PASS",
        "schema_version": 1,
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "source_dataset": str(source.resolve()),
        "source_dataset_sha256": sha256(source),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path),
        "record_count": records,
        "horizon": int(horizon),
        "derived_horizons": [int(horizon)],
        "derived_from_max_horizon": int(source_manifest["horizon"]),
    }
    for field in (
        "gmt_checkpoint_sha256",
        "config_sha256",
        "source_root",
        "source_commit",
        "counterfactual_engine",
        "counterfactual_engine_version",
        "state_schema_version",
        "utility_definition",
        "cache_sha256",
        "perception_cache_index_sha256",
        "annotations_sha256",
        "trace_sha256",
        "association_backend",
        "formal_gmt_association_adapter",
        "prelock",
        "selection_authority",
        "official_result_authority",
        "official_test_generation_authorized",
        "official_test_lock_sha256",
        "official_test_selection_protocol_sha256",
    ):
        if field in source_manifest:
            manifest[field] = source_manifest[field]
    Path(str(output) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.input.resolve(), args.output.resolve(), args.horizon), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
