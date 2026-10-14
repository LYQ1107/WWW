"""Build a compact, read-only audit of the master JEV evidence state.

The audit reads the already published Phase XVII--XXI evidence and the current
bounded recovery package.  It never edits checkpoints, labels, registries, or
historical worktrees.  Paths are configurable because the historical evidence
is intentionally kept outside this recovery checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object JSON: {path}")
    return value


def evidence(path: Path, status: Any = None, **extra: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "SHA256": sha256(path),
    }
    if status is not None:
        item["status"] = status
    item.update(extra)
    return item


def git_state(path: Path) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "-C", str(path), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    branch = run("symbolic-ref", "--short", "HEAD") or run("branch", "--show-current")
    return {
        "path": str(path),
        "HEAD": run("rev-parse", "HEAD"),
        "branch": branch,
        "clean": status == "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--phase21-root",
        type=Path,
        default=Path("/home/liuyeqiang/WWW_jev_phase21"),
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("/data1/liuyeqiang/WWW_jev_recovery_runtime/20261012_v1"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    phase21 = args.phase21_root.resolve()
    runtime = args.runtime.resolve()
    reports = repo / "reports" / "JEV_RECOVERY"
    docs = repo / "docs" / "JEV_RECOVERY"

    phase17 = phase21 / "reports" / "JEV_PHASE17"
    phase18 = phase21 / "reports" / "JEV_PHASE18"
    phase19 = phase21 / "reports" / "JEV_PHASE19"
    phase20 = phase21 / "reports" / "JEV_PHASE20"
    phase21_reports = phase21 / "reports" / "JEV_PHASE21"

    phase17_who = load_json(phase17 / "WHO_Q4_FACTORIAL.json")
    phase17_prefix = load_json(phase17 / "PREFIX_WHO_Q4_FACTORIAL.json")
    phase17_owner = load_json(phase17 / "OWNER_IDENTIFIABILITY.json")
    phase17_final = load_json(phase17 / "FINAL_GO_NO_GO.json")
    phase17_soft = load_json(phase17 / "SOFT_MEMORY_CAUSAL_RESULTS.json")
    phase18_final = load_json(phase18 / "FINAL_GO_NO_GO.json")
    phase19_final = load_json(phase19 / "FINAL_GO_NO_GO.json")
    phase20_final = load_json(phase20 / "FINAL_GO_NO_GO.json")
    phase21_final = load_json(phase21_reports / "FINAL_GO_NO_GO.json")
    recovery_review = load_json(reports / "RECOVERY_FINAL_REVIEW_20261014.json")
    ready = load_json(runtime / "finish" / "READY.json")
    monitor = load_json(runtime / "live_monitor" / "STATUS.json")
    full_online = load_json(reports / "FULL_ONLINE_RESULTS.json")
    publication = load_json(runtime / "publication" / "LAST_VERIFIED.json")

    formal = [
        row
        for row in full_online.get("summary", [])
        if row.get("track") in {"R2_XIV", "R3_XIII"}
        and row.get("status") == "COMPLETE_THREE_SEEDS"
    ]

    requirements = {
        "phase17_source_audit": {
            "judgment": "COMPLETE_SOURCE_INSPECTION_RUNTIME_PARITY_SEPARATE",
            "required_scope": "six-source implementation, IO, training and cache audit",
            "evidence": [
                evidence(phase21 / "docs" / "PHASE17_VISUAL_JEV_SOURCE_AUDIT.md"),
                evidence(phase17 / "EXTERNAL_CODE_MANIFEST.json"),
            ],
            "limit": "source inspection does not prove external runtime portability or identity performance",
        },
        "who_q4_native_causal_attribution": {
            "judgment": "COMPLETE_FROZEN_DIAGNOSTIC_NOT_NEW_POLICY",
            "required_scope": "native WHO/Q4 cold factorial, prefix feedback and first divergence",
            "status": phase17_who.get("status"),
            "evidence": [
                evidence(phase17 / "WHO_Q4_FACTORIAL.json", phase17_who.get("status")),
                evidence(phase17 / "PREFIX_WHO_Q4_FACTORIAL.json", phase17_prefix.get("status")),
                evidence(phase17 / "FIRST_DIVERGENCE.json"),
            ],
            "limit": "factorial interaction is an attribution contrast, not a unique physical module share",
        },
        "owner_identifiability": {
            "judgment": "COMPLETE_BOUNDED_CENSUS_TRAINING_NOT_QUALIFIED",
            "status": phase17_owner.get("status"),
            "independent_difficult_groups": phase17_owner.get("independent_difficult_groups"),
            "evidence": [
                evidence(phase17 / "OWNER_IDENTIFIABILITY.json", phase17_owner.get("status")),
                evidence(phase17 / "PHASE17_OWNER_EVIDENCE_AUDIT.json"),
                evidence(phase19 / "LEARNING_ELIGIBILITY.json"),
            ],
            "limit": "first past-certified native-ID/person binding is not a true birth-owner or future-safe certificate",
        },
        "visual_and_soft_memory_diagnostics": {
            "judgment": "COMPLETE_BOUNDED_NEGATIVE_DIAGNOSTICS",
            "phase17_soft_memory_status": phase17_soft.get("status"),
            "phase21_visual_status": phase21_final.get("status"),
            "evidence": [
                evidence(phase17 / "SOFT_MEMORY_CAUSAL_RESULTS.json", phase17_soft.get("status")),
                evidence(phase17 / "VISUAL_JEV_COMPARISON.json"),
                evidence(phase21_reports / "VISUAL_EXECUTION_PROFILE_PARITY.json"),
                evidence(phase21_reports / "VISUAL_PROFILE_ATTRIBUTION.json"),
                evidence(phase21_reports / "VISUAL_EXECUTION_LATENCY.json"),
            ],
            "limit": "no qualified learned memory admission, native end-to-end visual deployment, or independent visual identity gain",
        },
        "qualified_training": {
            "judgment": "NOT_QUALIFIED",
            "evidence": [
                evidence(phase17 / "LEARNING_ELIGIBILITY.json"),
                evidence(phase18 / "LEARNING_ELIGIBILITY.json"),
                evidence(phase19 / "LEARNING_ELIGIBILITY.json"),
            ],
            "limit": "independent safety/ownership gates remain below threshold; new lifecycle training remains NOT_RUN",
        },
        "independent_evaluation": {
            "judgment": "NOT_ESTABLISHED",
            "evidence": [
                evidence(phase17 / "FINAL_GO_NO_GO.json", phase17_final.get("status")),
                evidence(phase18 / "FINAL_GO_NO_GO.json", phase18_final.get("status")),
                evidence(phase19 / "FINAL_GO_NO_GO.json", phase19_final.get("status")),
                evidence(phase20 / "FINAL_GO_NO_GO.json", phase20_final.get("status")),
                evidence(phase21_reports / "FINAL_GO_NO_GO.json", phase21_final.get("status")),
                evidence(reports / "RECOVERY_FINAL_REVIEW_20261014.json", recovery_review.get("status")),
            ],
            "limit": "no official TEST, Full24, or sealed 20/21/22 evaluation is authorized",
        },
        "r2_r3_monitor_and_formal_routes": {
            "judgment": "COMPLETE_BOUNDED_EXECUTION",
            "formal_three_seed_entries": len(formal),
            "evidence": [
                evidence(runtime / "live_monitor" / "STATUS.json", monitor.get("status")),
                evidence(runtime / "finish" / "READY.json", ready.get("MASTER_BREAKTHROUGH")),
                evidence(reports / "FULL_ONLINE_RESULTS.json", full_online.get("status")),
                evidence(reports / "RECOVERY_FINAL_REVIEW_20261014.json", recovery_review.get("status")),
            ],
            "live_jobs": monitor.get("live_jobs"),
            "unreviewed_failures": monitor.get("unreviewed_failures"),
        },
        "frozen_evidence_and_publication": {
            "judgment": "VERIFIED",
            "evidence": [
                evidence(reports / "ARTIFACT_SHA_MANIFEST.json"),
                evidence(reports / "RECOVERY_FINAL_REVIEW_20261014.json"),
                evidence(runtime / "publication" / "LAST_VERIFIED.json", publication.get("remote_HEAD")),
            ],
            "remote_HEAD": publication.get("remote_HEAD"),
            "branch": publication.get("branch"),
            "sealed20_21_22": recovery_review.get("scope", {}).get("sealed20_21_22"),
            "official_TEST": False,
        },
    }

    result: Dict[str, Any] = {
        "status": "COMPLETE_EVIDENCE_AUDIT_MASTER_NOT_COMPLETE",
        "captured_UTC": datetime.now(timezone.utc).isoformat(),
        "master_complete": False,
        "scientific_success": False,
        "current_recovery_decision": recovery_review.get("decision"),
        "current_recovery_commit": publication.get("remote_HEAD"),
        "worktrees": {
            "recovery": git_state(repo),
            "phase21_evidence": git_state(phase21),
        },
        "requirements": requirements,
        "monitor": monitor,
        "next_action": "Require a new preregistered scope and independently valid ownership/causal supervision before any further lifecycle training; keep current MATCH NO_GO frozen.",
    }

    out_json = (args.output_json or (reports / "MASTER_EVIDENCE_AUDIT_20261014.json")).resolve()
    out_md = (args.output_md or (docs / "MASTER_EVIDENCE_AUDIT_20261014.md")).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# JEV master evidence audit (2026-10-14)",
        "",
        "This is a read-only index of the published Phase XVII–XXI evidence and the current bounded recovery package. It does not convert a phase result into master scientific success.",
        "",
        f"- Audit status: `{result['status']}`",
        f"- Current recovery decision: `{result['current_recovery_decision']}`",
        f"- Recovery remote HEAD: `{result['current_recovery_commit']}`",
        f"- R2/R3 formal three-seed entries: `{len(formal)}`",
        f"- Live jobs: `{monitor.get('live_jobs')}`",
        f"- Unreviewed failures: `{monitor.get('unreviewed_failures')}`",
        "",
        "| Requirement | Current judgment | Evidence boundary |",
        "|---|---|---|",
    ]
    for key, item in requirements.items():
        boundary = item.get("limit", "See evidence entries")
        lines.append(f"| `{key}` | `{item['judgment']}` | {boundary} |")
    lines.extend(
        [
            "",
            "The auditable route is complete as a bounded negative/diagnostic package. Qualified lifecycle training, independent evaluation, native visual deployment, and a unified MATCH/REACT/WRITE scientific contribution remain unestablished. Any continuation requires a new preregistered scope; no sealed data or historical checkpoint is unlocked by this audit.",
            "",
            "The machine-readable file records the exact paths and SHA-256 values used for each requirement.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "json": str(out_json), "markdown": str(out_md)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
