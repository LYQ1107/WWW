"""Verify that the bounded R2/R3 recovery queue is terminal.

This is a read-only audit.  It reads the published experiment registry, the
runtime monitor, READY marker, and formal online summary; it does not restart
workers, alter checkpoints, or infer scientific success from completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "present": path.is_file(), "SHA256": sha256(path)}


def load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object JSON: {path}")
    return value


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
    ready_path = runtime / "finish" / "READY.json"
    online_path = reports / "FULL_ONLINE_RESULTS.json"
    review_path = reports / "RECOVERY_FINAL_REVIEW_20261014.json"
    failure_review_path = reports / "R2_FUSION_FAILURE_CLASSIFICATION_20261014.json"
    registry = load(registry_path)
    monitor = load(monitor_path)
    ready = load(ready_path)
    online = load(online_path)
    review = load(review_path)
    failure_review = load(failure_review_path) if failure_review_path.is_file() else None

    jobs = registry.get("jobs", {})
    if not isinstance(jobs, dict):
        raise TypeError("registry.jobs must be an object")
    status_counts = Counter(
        item.get("status") for item in jobs.values() if isinstance(item, dict)
    )
    failed = sorted(
        name
        for name, item in jobs.items()
        if isinstance(item, dict) and item.get("status") == "FAILED"
    )
    r2_r3 = {
        name: item
        for name, item in jobs.items()
        if name.startswith("R2_") or name.startswith("R3_")
    }
    r2_r3_counts = Counter(
        item.get("status") for item in r2_r3.values() if isinstance(item, dict)
    )
    formal = [
        row
        for row in online.get("summary", [])
        if row.get("track") in {"R2_XIV", "R3_XIII"}
        and row.get("status") == "COMPLETE_THREE_SEEDS"
    ]
    live_jobs = monitor.get("live_jobs") or []
    unreviewed_failures = monitor.get("unreviewed_failures") or []
    terminal_statuses = {"COMPLETE", "FAILED"}
    all_jobs_terminal = bool(jobs) and all(
        isinstance(item, dict) and item.get("status") in terminal_statuses
        for item in jobs.values()
    )
    queue_terminal = (
        all_jobs_terminal
        and not live_jobs
        and not unreviewed_failures
        and monitor.get("automatic_restart") is False
    )

    result: Dict[str, Any] = {
        "status": (
            "COMPLETE_BOUNDED_R2_R3_QUEUE_TERMINAL"
            if queue_terminal
            else "QUEUE_TERMINALITY_UNCONFIRMED"
        ),
        "captured_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "R2/R3 bounded recovery queue terminal-state audit",
        "queue_terminal": queue_terminal,
        "scientific_success_inferred": False,
        "master_goal_complete": False,
        "registry": {
            "path": str(registry_path),
            "status": registry.get("status"),
            "total_jobs": len(jobs),
            "status_counts": dict(sorted(status_counts.items(), key=lambda pair: str(pair[0]))),
            "failed_jobs": failed,
            "all_jobs_terminal": all_jobs_terminal,
            "R2_R3_job_count": len(r2_r3),
            "R2_R3_status_counts": dict(
                sorted(r2_r3_counts.items(), key=lambda pair: str(pair[0]))
            ),
            "SHA256": sha256(registry_path),
        },
        "monitor": {
            "path": str(monitor_path),
            "status": monitor.get("status"),
            "live_jobs": live_jobs,
            "unreviewed_failures": unreviewed_failures,
            "automatic_restart": monitor.get("automatic_restart"),
            "scientific_success_not_inferred": monitor.get(
                "scientific_success_not_inferred"
            ),
            "SHA256": sha256(monitor_path),
        },
        "ready_marker": {
            "path": str(ready_path),
            "MASTER_BREAKTHROUGH": ready.get("MASTER_BREAKTHROUGH"),
            "legacy_goal_gate_unchanged": ready.get("legacy_goal_gate_unchanged"),
            "SHA256": sha256(ready_path),
        },
        "formal_online_summary": {
            "path": str(online_path),
            "formal_R2_R3_three_seed_entries": len(formal),
            "SHA256": sha256(online_path),
        },
        "scientific_review": {
            "path": str(review_path),
            "decision": review.get("decision"),
            "development_status": review.get("development_status"),
            "SHA256": sha256(review_path),
        },
        "failure_review": {
            "path": str(failure_review_path),
            "present": failure_review is not None,
            "status": failure_review.get("status") if failure_review else None,
            "failures": [
                item.get("registry_name")
                for item in (failure_review or {}).get("failures", [])
            ],
            "SHA256": sha256(failure_review_path),
        },
        "failure_interpretation": (
            "The two FAILED registry entries remain preserved with their logs; "
            "they are not silently converted to success and are excluded from "
            "the claim that all attempts completed. The separate failure review "
            "classifies them as engineering input-contract failures and records "
            "the completed v2/v3 successors."
        ),
        "next_action": (
            "Do not restart the bounded queue. Preserve the reviewed NO_GO package; "
            "any new policy or independent-data experiment needs a new preregistered "
            "scope and valid supervision."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# R2/R3 bounded queue terminal audit (2026-10-14)",
        "",
        "This is a read-only process and evidence audit. It does not restart the queue or turn execution completion into a scientific success claim.",
        "",
        f"- Status: `{result['status']}`",
        f"- Registered jobs: `{len(jobs)}`; status counts: `{dict(status_counts)}`",
        f"- R2/R3 jobs: `{len(r2_r3)}`; status counts: `{dict(r2_r3_counts)}`",
        f"- Live jobs: `{live_jobs}`",
        f"- Unreviewed failures: `{unreviewed_failures}`",
        f"- Automatic restart: `{monitor.get('automatic_restart')}`",
        f"- Formal R2/R3 three-seed entries: `{len(formal)}`",
        f"- READY.MASTER_BREAKTHROUGH: `{ready.get('MASTER_BREAKTHROUGH')}`",
        f"- Scientific success inferred: `false`",
        "",
        "Two failed registry entries are retained with their logs and are explicitly classified in the separate failure review; the terminal claim means there are no pending or live jobs and no unreviewed failure, not that every attempt succeeded.",
        "",
        "The next action is to preserve the reviewed NO_GO package. Any new training or independent evaluation requires a new preregistered scope and valid supervision.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "total_jobs": len(jobs),
                "failed_jobs": failed,
                "live_jobs": live_jobs,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
