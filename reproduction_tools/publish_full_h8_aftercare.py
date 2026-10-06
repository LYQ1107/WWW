"""Publish a small, provenance-bound Full H=8 postprocess summary.

The canonical records remain in the runtime data volume.  This publisher only
copies machine-readable manifests, hashes, and the already aggregated
single-seed result into the repository so the research state is reviewable on
GitHub without committing the 1.1M-record dataset.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jev_full_h8_authorization import read_formal_authorization


SEED = 20261003
METHODS = ("question_threshold", "question_conditioned_mlp", "jev")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def artifact(path: Path, *, required: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return result
    result["sha256"] = sha256(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--formal-gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    runtime = args.runtime_root.resolve()
    repo = args.repo_root.resolve()
    formal = read_formal_authorization(args.formal_gate_report.resolve())
    partition_path = runtime / "partition" / "partition_manifest.json"
    records_path = runtime / "FULL_H8_FORMAL.records.jsonl"
    records_manifest_path = runtime / "FULL_H8_FORMAL.records.jsonl.manifest.json"
    compact_manifest_path = runtime / "FULL_H8_POLICY_COMPACT" / "manifest.json"
    split_path = runtime / "POLICY_SPLIT.json"
    postprocess_path = runtime / "POSTPROCESS_MANIFEST.json"
    first_round_path = runtime / "FIRST_ROUND_REPORT.json"
    first_round_markdown_path = runtime / "FIRST_ROUND_REPORT.md"

    partition = read_json(partition_path)
    records_manifest = read_json(records_manifest_path)
    compact = read_json(compact_manifest_path)
    split = read_json(split_path)
    postprocess = read_json(postprocess_path)
    first_round = read_json(first_round_path)
    expected_records = int(partition.get("total_main_decisions", -1))
    failures: list[str] = []
    if partition.get("status") != "COMPLETE" or len(partition.get("videos") or {}) != 24:
        failures.append("partition_not_24_video_complete")
    if records_manifest.get("status") != "PASS" or int(records_manifest.get("record_count", -1)) != expected_records:
        failures.append("final_records_manifest_not_matching_pass")
    if not records_path.is_file() or sum(1 for line in records_path.open(encoding="utf-8") if line.strip()) != expected_records:
        failures.append("final_records_line_count_mismatch")
    if compact.get("status") != "PASS" or int(compact.get("records", -1)) != expected_records:
        failures.append("compact_manifest_not_matching_pass")
    if int(split.get("seed", -1)) != SEED or split.get("official_test_used_for_search") is not False:
        failures.append("policy_split_not_locked_train_only_seed")
    if postprocess.get("status") != "PASS":
        failures.append("postprocess_manifest_not_pass")
    if first_round.get("status") != "PASS" or list(first_round.get("seeds", ())) != [SEED]:
        failures.append("first_round_not_single_seed_pass")
    if failures:
        raise RuntimeError("cannot publish incomplete Full H=8 aftercare: " + ", ".join(failures))

    method_manifests = {}
    for method in METHODS:
        path = runtime / "FIRST_ROUND_METHODS" / method / "method_manifest.json"
        manifest = read_json(path)
        if manifest.get("status") != "PASS" or list(manifest.get("seeds", ())) != [SEED]:
            raise RuntimeError(f"method manifest is not a locked PASS: {path}")
        method_manifests[method] = {
            "status": manifest.get("status"),
            "method_label": manifest.get("method_label"),
            "dataset": manifest.get("dataset"),
            "policy_split": manifest.get("policy_split"),
            "seeds": manifest.get("seeds"),
            "training": manifest.get("training"),
            "artifact": artifact(path),
        }

    report = {
        "schema_version": "jev_full_h8_aftercare_publication_v1",
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "FULL_H8_CURRENT_HEAD_POSTPROCESS_SUMMARY_NOT_FINAL_TEST_TRACKING_RESULT",
        "seed": SEED,
        "formal_gate": {
            "status": formal.get("status"),
            "report": artifact(args.formal_gate_report.resolve()),
            "final_gate": formal.get("final_gate"),
            "single_records": formal.get("single_records"),
            "chunked_records": formal.get("chunked_records"),
            "canonical_sha_identical": formal.get("canonical_sha_identical"),
        },
        "partition": {
            "manifest": artifact(partition_path),
            "status": partition.get("status"),
            "videos": len(partition.get("videos") or {}),
            "records": expected_records,
            "trace": partition.get("trace"),
            "cache": partition.get("cache"),
        },
        "final_records": {
            "records": artifact(records_path),
            "manifest": artifact(records_manifest_path),
        },
        "compact": {
            "manifest": artifact(compact_manifest_path),
            "records": compact.get("records"),
            "state_dim": compact.get("state_dim"),
        },
        "policy_split": {
            "artifact": artifact(split_path),
            "seed": split.get("seed"),
            "official_test_used_for_search": split.get("official_test_used_for_search"),
        },
        "postprocess": {
            "artifact": artifact(postprocess_path),
            "status": postprocess.get("status"),
        },
        "first_round": {
            "report": artifact(first_round_path),
            "markdown": artifact(first_round_markdown_path),
            "status": first_round.get("status"),
            "seeds": first_round.get("seeds"),
            "jev_beats_both_on_val_utility": first_round.get("jev_beats_both_on_val_utility"),
            "result": first_round,
            "methods": method_manifests,
        },
        "official_test_used": False,
        "next_step": "Run formal closed-loop tracking/evaluation only after checking runtime wrapper parity and controller/runtime provenance.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = "\n".join(
        [
            "# Full H=8 aftercare publication",
            "",
            "> This is a provenance/postprocess summary, not the final TEST tracking result.",
            "",
            f"- Status: `{report['status']}`",
            f"- Videos: `{report['partition']['videos']}`",
            f"- Canonical records: `{report['partition']['records']}`",
            f"- Seed: `{SEED}`",
            f"- Formal canonical SHA identical: `{report['formal_gate']['canonical_sha_identical']}`",
            f"- JEV beats both on validation utility: `{report['first_round']['jev_beats_both_on_val_utility']}`",
            "- Official TEST data used for search: `False`",
            "",
            "The full records stay in the runtime data volume. The JSON report records their SHA-256 and the SHA-256 of every published manifest.",
            "",
            "Next: verify full runtime feature/candidate parity, then run the locked closed-loop tracking comparison.",
            "",
        ]
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": report["status"], "records": expected_records, "videos": 24}, indent=2))


if __name__ == "__main__":
    main()
