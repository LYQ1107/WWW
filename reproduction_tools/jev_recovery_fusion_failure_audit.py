"""Classify preserved supplementary-fuser failures without restarting them.

The R2 supplementary final-fuser queue had two early engineering attempts that
failed before producing a usable result.  Later frozen reruns repaired the
input routing/schema and completed the corresponding diagnostics.  This audit
keeps the failed registry entries immutable, binds their logs and preregistration,
and states exactly which result claims are and are not supported.

It is deliberately read-only: no process is started, no registry status is
rewritten, and no checkpoint, label, or gate is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "present": path.is_file(), "SHA256": sha256(path)}


def load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def first_line_matching(text: str, pattern: str) -> Optional[str]:
    rx = re.compile(pattern)
    for line in text.splitlines():
        if rx.search(line):
            return line.strip()
    return None


def result_refs(reports: Path, kind: str) -> List[Dict[str, Any]]:
    """Return completed diagnostic artifacts for one fuser kind."""
    paths: List[Path] = []
    for root in (reports / "diagnostic_fusion", reports / "diagnostic_fusion_native"):
        if root.is_dir():
            paths.extend(sorted(root.glob(f"*{kind}*")))
            paths.extend(sorted(root.glob(f"**/*{kind}*")))
    unique: List[Path] = []
    seen = set()
    for path in paths:
        if path.is_file() and str(path) not in seen:
            seen.add(str(path))
            unique.append(path)
    return [ref(path) for path in unique]


def classify(
    name: str, job: Dict[str, Any], log_path: Path, log_text: str, jobs: Dict[str, Any], reports: Path
) -> Dict[str, Any]:
    if name == "R2_FUSION_explicit_scores_identity_v2":
        missing = first_line_matching(log_text, r"FileNotFoundError:.*RESULT\.json")
        classification = "INPUT_ARTIFACT_MISSING"
        cause = "identity helper was pointed at a non-existent parent trajectory output"
        successor = "R2_FUSION_explicit_scores_identity_v3"
        successor_note = "v3 completed after the frozen identity-source routing correction"
        missing_artifact = (
            "/data1/liuyeqiang/WWW_jev_recovery_runtime/20261012_v1/"
            "diagnostic_fusion_native_v2/online_v1/pilot/parent/seed20261009/video16/RESULT.json"
        )
        details = {"missing_artifact": missing_artifact, "exception_line": missing}
    elif name == "R2_FUSION_explicit_scores_tiny_v1":
        key = first_line_matching(log_text, r"KeyError: ['\"]anchor_visual['\"]")
        classification = "INPUT_SCHEMA_MISMATCH"
        cause = "the Tiny fixture lacked the anchor_visual field expected by the diagnostic model"
        successor = "R2_FUSION_explicit_scores_tiny_v2"
        successor_note = "v2 completed using the repaired Tiny fixture/schema"
        details = {"missing_field": "anchor_visual", "exception_line": key}
    else:
        classification = "UNCLASSIFIED_ENGINEERING_FAILURE"
        cause = "not covered by the frozen two-entry classification contract"
        successor = None
        successor_note = None
        details = {}

    successor_job = jobs.get(successor) if successor else None
    return {
        "registry_name": name,
        "registry_status": job.get("status"),
        "exit_code": job.get("exit_code"),
        "pid_recorded": job.get("PID"),
        "command": job.get("command"),
        "classification": classification,
        "cause": cause,
        "details": details,
        "log": ref(log_path),
        "successor": {
            "registry_name": successor,
            "status": successor_job.get("status") if isinstance(successor_job, dict) else None,
            "note": successor_note,
        },
        "completed_artifacts_for_kind": result_refs(reports, "explicit_scores"),
        "rerun_authorized_in_current_queue": False,
        "claim_boundary": (
            "This failed attempt is retained as an engineering failure. Its existence does not "
            "invalidate completed formal R2/R3 routes, but the failed attempt itself cannot be "
            "counted as a successful fuser result."
        ),
    }


def render_markdown(result: Dict[str, Any]) -> str:
    lines = [
        "# R2 supplementary fuser failure classification (2026-10-14)",
        "",
        "This is a read-only audit of two preserved FAILED registry entries. It did not restart a process, rewrite a registry status, alter a checkpoint or label, or change a gate.",
        "",
        f"- Status: `{result['status']}`",
        f"- Queue terminal: `{result['queue_terminal']}`",
        f"- Live jobs: `{result['live_jobs']}`",
        f"- Automatic restart: `{result['automatic_restart']}`",
        f"- Preregistration stop rule: `{result['preregistration_stop_rule']}`",
        "",
        "| Preserved failed entry | Classification | Corrected successor | Result boundary |",
        "|---|---|---|---|",
    ]
    for item in result["failures"]:
        successor = item["successor"]
        lines.append(
            f"| `{item['registry_name']}` | `{item['classification']}` | "
            f"`{successor['registry_name']}` (`{successor['status']}`) | "
            "failed attempt remains excluded from success claims |"
        )
    lines.extend(
        [
            "",
            "The identity failure is a missing parent RESULT routing artifact. The Tiny failure is a schema contract mismatch (`anchor_visual`). Both failures occurred before a valid final comparison for that attempt; later v2/v3 entries completed the corresponding frozen diagnostics and are the artifacts used by the bounded fusion result index.",
            "",
            "The corrected successors do not erase the earlier failures. The formal claim is therefore: the bounded R2/R3 result package is terminal and the four supplementary fuser diagnostics have usable completed artifacts, while the two earlier engineering attempts remain FAILED and are not evidence of scientific success or failure of the JEV hypothesis.",
            "",
            "No automatic retry is authorized under the current frozen queue. A further rerun would require a new preregistered execution identity and an explicit input-contract repair.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("/data1/liuyeqiang/WWW_jev_recovery_runtime/20261012_v1"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    runtime = args.runtime.resolve()
    reports = repo / "reports" / "JEV_RECOVERY"
    registry_path = reports / "EXPERIMENT_REGISTRY.json"
    monitor_path = runtime / "live_monitor" / "STATUS.json"
    prereg_path = reports / "DIAGNOSTIC_FUSION_PREREGISTRATION.json"
    index_path = reports / "DIAGNOSTIC_FUSION_RESULT_INDEX.json"
    registry = load(registry_path)
    jobs = registry.get("jobs", {})
    if not isinstance(jobs, dict):
        raise TypeError("registry.jobs must be an object")
    monitor = load(monitor_path)
    prereg = load(prereg_path)
    index = load(index_path)

    target_names = [
        "R2_FUSION_explicit_scores_identity_v2",
        "R2_FUSION_explicit_scores_tiny_v1",
    ]
    failures: List[Dict[str, Any]] = []
    for name in target_names:
        job = jobs.get(name)
        if not isinstance(job, dict) or job.get("status") != "FAILED":
            raise AssertionError(f"expected preserved FAILED registry entry: {name}")
        log_path = Path(job.get("log", {}).get("path", ""))
        failures.append(classify(name, job, log_path, read_text(log_path), jobs, reports))

    all_jobs_terminal = bool(jobs) and all(
        isinstance(job, dict) and job.get("status") in {"COMPLETE", "FAILED"}
        for job in jobs.values()
    )
    queue_terminal = (
        all_jobs_terminal
        and not monitor.get("live_jobs")
        and not monitor.get("unreviewed_failures")
        and monitor.get("automatic_restart") is False
    )
    result: Dict[str, Any] = {
        "status": "COMPLETE_FAILURE_REVIEW_WITH_NO_RESTART",
        "captured_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "Preserved R2 supplementary fuser engineering failures and corrected successors",
        "queue_terminal": queue_terminal,
        "live_jobs": monitor.get("live_jobs") or [],
        "automatic_restart": monitor.get("automatic_restart"),
        "scientific_success_inferred": False,
        "registry": ref(registry_path),
        "monitor": ref(monitor_path),
        "preregistration": ref(prereg_path),
        "preregistration_stop_rule": prereg.get("stop_rule"),
        "result_index": {
            **ref(index_path),
            "status": index.get("status"),
            "independent_validation": index.get("independent_validation"),
            "original_R2_R3_gates_changed": index.get("original_R2_R3_gates_changed"),
        },
        "failures": failures,
        "formal_routes_affected": False,
        "supplementary_fuser_claim": "COMPLETED_CORRECTED_ARTIFACTS_WITH_PRESERVED_EARLY_ENGINEERING_FAILURES",
        "rerun_policy": "NO_RERUN_IN_CURRENT_FROZEN_QUEUE",
        "conclusion": (
            "The two FAILED entries are input-contract/routing failures, not hidden scientific "
            "results. Their corrected v2/v3 successors completed the bounded supplementary "
            "diagnostics. Keep the failed entries and logs; do not restart them automatically."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "queue_terminal": queue_terminal, "failures": target_names}, ensure_ascii=False))


if __name__ == "__main__":
    main()
