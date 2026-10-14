"""Requirement-by-requirement audit of the evidence-grounded master goal.

The report distinguishes completed bounded audits from the gates that remain
unqualified.  It is read-only and never changes checkpoints, labels, queues,
or historical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def load(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object JSON: {path}")
    return value


def evidence(path: Path, **extra: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "SHA256": sha256(path),
    }
    item.update(extra)
    return item


def item(
    status: str,
    proof: str,
    limitation: str,
    evidence_items: Iterable[Dict[str, Any]],
    **extra: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": status,
        "proof": proof,
        "limitation": limitation,
        "evidence": list(evidence_items),
    }
    result.update(extra)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--phase21-root", type=Path, default=Path("/home/liuyeqiang/WWW_jev_phase21")
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("/data1/liuyeqiang/WWW_jev_recovery_runtime/20261012_v1"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    phase21 = args.phase21_root.resolve()
    runtime = args.runtime.resolve()
    recovery_reports = repo / "reports" / "JEV_RECOVERY"
    recovery_docs = repo / "docs" / "JEV_RECOVERY"
    p17 = phase21 / "reports" / "JEV_PHASE17"
    p18 = phase21 / "reports" / "JEV_PHASE18"
    p19 = phase21 / "reports" / "JEV_PHASE19"
    p20 = phase21 / "reports" / "JEV_PHASE20"
    p21 = phase21 / "reports" / "JEV_PHASE21"

    p17_who = load(p17 / "WHO_Q4_FACTORIAL.json")
    p17_prefix = load(p17 / "PREFIX_WHO_Q4_FACTORIAL.json")
    p17_owner = load(p17 / "OWNER_IDENTIFIABILITY.json")
    p17_owner_audit = load(p17 / "PHASE17_OWNER_EVIDENCE_AUDIT.json")
    p17_soft = load(p17 / "SOFT_MEMORY_CAUSAL_RESULTS.json")
    p17_final = load(p17 / "FINAL_GO_NO_GO.json")
    p18_final = load(p18 / "FINAL_GO_NO_GO.json")
    p19_final = load(p19 / "FINAL_GO_NO_GO.json")
    p20_final = load(p20 / "FINAL_GO_NO_GO.json")
    p21_final = load(p21 / "FINAL_GO_NO_GO.json")
    review = load(recovery_reports / "RECOVERY_FINAL_REVIEW_20261014.json")
    master_goal = load(recovery_reports / "MASTER_GOAL.json")
    queue = load(recovery_reports / "R2_R3_QUEUE_TERMINAL_AUDIT_20261014.json")
    access = load(recovery_reports / "NEXT_SCOPE_ACCESS_AUDIT_20261014.json")
    candidate = load(recovery_reports / "NEXT_SCOPE_CANDIDATE_COPY_AUDIT_20261014.json")
    source_candidates = load(recovery_reports / "NEXT_SCOPE_SOURCE_CANDIDATE_AUDIT_20261014.json")
    monitor = load(runtime / "live_monitor" / "STATUS.json")
    ready = load(runtime / "finish" / "READY.json")
    publication = load(runtime / "publication" / "LAST_VERIFIED.json")
    manifest = load(recovery_reports / "ARTIFACT_SHA_MANIFEST.json")

    requirements: Dict[str, Dict[str, Any]] = {}
    requirements["phase17_source_audit"] = item(
        "VERIFIED_BOUNDED",
        "Phase XVII source inspection artifact and external code manifest are present.",
        "Source inspection does not prove external runtime portability or identity performance.",
        [
            evidence(phase21 / "docs" / "PHASE17_VISUAL_JEV_SOURCE_AUDIT.md"),
            evidence(phase21 / "reports" / "JEV_PHASE17" / "EXTERNAL_CODE_MANIFEST.json"),
        ],
    )
    requirements["who_q4_native_causal_attribution"] = item(
        "VERIFIED_BOUNDED",
        "Cold-start and prefix WHO/Q4 factorial artifacts are complete.",
        "The factorial is a bounded attribution contrast, not a unique physical module share.",
        [
            evidence(p17 / "WHO_Q4_FACTORIAL.json", status=p17_who.get("status")),
            evidence(p17 / "PREFIX_WHO_Q4_FACTORIAL.json", status=p17_prefix.get("status")),
            evidence(p17 / "FIRST_DIVERGENCE.json"),
        ],
        cold_start_status=p17_who.get("status"),
        prefix_status=p17_prefix.get("status"),
    )
    requirements["owner_identifiability"] = item(
        "VERIFIED_BOUNDED_NOT_TRAINING_QUALIFICATION",
        "Owner census and evidence audit completed under the frozen prefix protocol.",
        "A bounded past-certified owner is not a true birth-owner or future-safe certificate.",
        [
            evidence(p17 / "OWNER_IDENTIFIABILITY.json", status=p17_owner.get("status")),
            evidence(p17 / "PHASE17_OWNER_EVIDENCE_AUDIT.json", status=p17_owner_audit.get("status")),
            evidence(p19 / "LEARNING_ELIGIBILITY.json"),
        ],
        owner_status=p17_owner.get("status"),
        owner_audit_status=p17_owner_audit.get("status"),
    )
    requirements["visual_and_soft_memory_diagnostics"] = item(
        "VERIFIED_BOUNDED_NEGATIVE_DIAGNOSTICS",
        "Soft-memory safety and Phase XXI visual execution parity diagnostics are recorded.",
        "No qualified learned memory admission or independent visual identity gain was established.",
        [
            evidence(p17 / "SOFT_MEMORY_CAUSAL_RESULTS.json", status=p17_soft.get("status")),
            evidence(p17 / "VISUAL_JEV_COMPARISON.json"),
            evidence(p21 / "VISUAL_EXECUTION_PROFILE_PARITY.json"),
            evidence(p21 / "VISUAL_PROFILE_ATTRIBUTION.json"),
            evidence(p21 / "VISUAL_EXECUTION_LATENCY.json"),
        ],
        soft_status=p17_soft.get("status"),
        phase21_status=p21_final.get("status"),
    )
    requirements["qualified_training"] = item(
        "NOT_QUALIFIED_GATE_PRESERVED",
        "Phase XVIII–XXI eligibility records keep new training out of scope.",
        "Independent owner/safety supervision remains below the preregistered threshold.",
        [
            evidence(p18 / "LEARNING_ELIGIBILITY.json"),
            evidence(p19 / "LEARNING_ELIGIBILITY.json"),
            evidence(p20 / "FINAL_GO_NO_GO.json", status=p20_final.get("status")),
            evidence(p21 / "FINAL_GO_NO_GO.json", status=p21_final.get("status")),
        ],
        new_training={
            "phase19": p19_final.get("new_training"),
            "phase20": p20_final.get("new_training"),
            "phase21": p21_final.get("new_training"),
        },
    )
    requirements["independent_evaluation"] = item(
        "NOT_ESTABLISHED_GATE_PRESERVED",
        "Review and phase records explicitly keep independent evaluation ineligible.",
        "No official TEST, Full24, or sealed 20/21/22 evaluation is authorized.",
        [
            evidence(recovery_reports / "RECOVERY_FINAL_REVIEW_20261014.json", status=review.get("status")),
            evidence(p17 / "FINAL_GO_NO_GO.json", status=p17_final.get("status")),
            evidence(p18 / "FINAL_GO_NO_GO.json", status=p18_final.get("status")),
            evidence(p19 / "FINAL_GO_NO_GO.json", status=p19_final.get("status")),
            evidence(p20 / "FINAL_GO_NO_GO.json", status=p20_final.get("status")),
            evidence(p21 / "FINAL_GO_NO_GO.json", status=p21_final.get("status")),
        ],
        official_TEST=review.get("scope", {}).get("official_TEST"),
        Full24=review.get("scope", {}).get("sealed20_21_22") is False,
        decision=review.get("decision"),
    )
    requirements["frozen_phase_v_xvi_evidence"] = item(
        "VERIFIED_PRESERVED",
        "The review records unchanged labels/weights, preserved prior negative evidence, and sealed later data.",
        "Preservation does not turn prior evidence into a positive JEV result.",
        [
            evidence(recovery_reports / "RECOVERY_FINAL_REVIEW_20261014.json", status=review.get("status")),
            evidence(p17 / "FINAL_GO_NO_GO.json", all_prior_negative_results_preserved=p17_final.get("all_prior_negative_results_preserved")),
            evidence(p17 / "FROZEN_PRIOR_INTEGRITY.json"),
        ],
        labels_modified=review.get("scope", {}).get("labels_modified"),
        weights_modified=review.get("scope", {}).get("weights_modified"),
        sealed20_21_22=review.get("scope", {}).get("sealed20_21_22"),
    )
    requirements["r2_r3_queue_monitor"] = item(
        "VERIFIED_TERMINAL_NO_SUCCESS_INFERENCE",
        "The registry, monitor, READY marker, and terminal queue audit agree on a terminal bounded queue.",
        "Two failed jobs remain failures; terminal execution is not scientific success.",
        [
            evidence(recovery_reports / "R2_R3_QUEUE_TERMINAL_AUDIT_20261014.json", status=queue.get("status")),
            evidence(runtime / "live_monitor" / "STATUS.json", status=monitor.get("status")),
            evidence(runtime / "finish" / "READY.json", MASTER_BREAKTHROUGH=ready.get("MASTER_BREAKTHROUGH")),
        ],
        queue_terminal=queue.get("queue_terminal"),
        live_jobs=monitor.get("live_jobs"),
        unreviewed_failures=monitor.get("unreviewed_failures"),
        registry_status_counts=queue.get("registry", {}).get("status_counts"),
    )
    remote_verified = publication.get("local_HEAD") == publication.get("remote_HEAD")
    requirements["reproducible_publication"] = item(
        "VERIFIED",
        "The publisher manifest and runtime publication marker verify the same local and remote HEAD.",
        "A published artifact is not evidence that the scientific hypothesis succeeded.",
        [
            evidence(recovery_reports / "ARTIFACT_SHA_MANIFEST.json", status=manifest.get("status")),
            evidence(runtime / "publication" / "LAST_VERIFIED.json", remote_HEAD=publication.get("remote_HEAD")),
        ],
        local_HEAD=publication.get("local_HEAD"),
        remote_HEAD=publication.get("remote_HEAD"),
        local_remote_equal=remote_verified,
    )
    requirements["independent_data_access_gate"] = item(
        "BLOCKING_GATE_AUTHORIZATION_UNRESOLVED",
        "The A-stage scan and candidate-copy metadata audit found a historical copy but no authorized path.",
        "No B-stage image/label use, training, or independent evaluation may proceed until legal access is recorded.",
        [
            evidence(recovery_reports / "NEXT_SCOPE_ACCESS_AUDIT_20261014.json", status=access.get("status")),
            evidence(recovery_reports / "NEXT_SCOPE_CANDIDATE_COPY_AUDIT_20261014.json", status=candidate.get("status")),
            evidence(recovery_reports / "NEXT_SCOPE_SOURCE_CANDIDATE_AUDIT_20261014.json", status=source_candidates.get("status")),
            evidence(p20 / "NEW_DATA_SOURCE_FEASIBILITY.json"),
        ],
        authorized_dataset_path_established=candidate.get("authorized_dataset_path_established"),
        authorization_status=candidate.get("authorization_status"),
        primary_candidate_count=source_candidates.get("primary_candidate_count"),
        source_candidate_status=source_candidates.get("status"),
    )

    positive_required = {
        "phase17_source_audit",
        "who_q4_native_causal_attribution",
        "owner_identifiability",
        "visual_and_soft_memory_diagnostics",
        "frozen_phase_v_xvi_evidence",
        "r2_r3_queue_monitor",
        "reproducible_publication",
    }
    positive_ok = all(
        requirements[key]["status"].startswith("VERIFIED") for key in positive_required
    )
    blocked_by_gates = any(
        requirements[key]["status"].startswith(("NOT_", "BLOCKING_"))
        for key in ("qualified_training", "independent_evaluation", "independent_data_access_gate")
    )
    result: Dict[str, Any] = {
        "status": "MASTER_GOAL_NOT_COMPLETE_REQUIREMENTS_AUDITED",
        "captured_UTC": datetime.now(timezone.utc).isoformat(),
        "master_goal_complete": False,
        "scientific_success": False,
        "positive_bounded_requirements_verified": positive_ok,
        "blocking_or_unmet_gates_present": blocked_by_gates,
        "requirements": requirements,
        "authoritative_master_goal": {
            "status": master_goal.get("status"),
            "scientific_success": master_goal.get("scientific_success"),
            "JEV_INDEPENDENT_VALUE": master_goal.get("JEV_INDEPENDENT_VALUE"),
        },
        "conclusion": (
            "All requested bounded audits and queue monitoring are evidenced, but the "
            "master goal is not complete: qualified training and independent evaluation "
            "remain gated, and independent data authorization is unresolved."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Master goal completion audit (2026-10-14)",
        "",
        "This is a requirement-by-requirement evidence audit. It does not equate phase completion, queue completion, or publication with scientific success.",
        "",
        f"- Overall status: `{result['status']}`",
        f"- Master goal complete: `{result['master_goal_complete']}`",
        f"- Scientific success: `{result['scientific_success']}`",
        f"- Positive bounded requirements verified: `{positive_ok}`",
        f"- Blocking/unmet gates present: `{blocked_by_gates}`",
        "",
        "| Requirement | Status | Boundary |",
        "|---|---|---|",
    ]
    for key, value in requirements.items():
        lines.append(f"| `{key}` | `{value['status']}` | {value['limitation']} |")
    lines.extend(
        [
            "",
            "The evidence supports a complete bounded audit package and a terminal R2/R3 queue. It does not support a qualified learned policy, independent validation, or an independent JEV advantage claim.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "master_goal_complete": False,
                "positive_bounded_requirements_verified": positive_ok,
                "blocking_or_unmet_gates_present": blocked_by_gates,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
