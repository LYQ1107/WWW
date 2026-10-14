"""Read-only audit for the next independent-data eligibility gate.

This deliberately does not download, open, or parse candidate dataset files.
It only records whether an authorized local path exists and binds the result to
the previously published Phase XX scope and source-feasibility documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TOKENS = ("mmptrack", "scout", "wildtrack")
SKIP_NAMES = {".git", ".cache", "anaconda3", "node_modules", "__pycache__"}


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_candidates(root: Path, max_depth: int) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if not root.is_dir():
        return found
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            lowered = name.lower()
            if any(token in lowered for token in TOKENS):
                try:
                    found.append(
                        {
                            "path": str(Path(entry.path)),
                            "is_dir": entry.is_dir(follow_symlinks=False),
                            "name": name,
                        }
                    )
                except OSError:
                    found.append({"path": str(Path(entry.path)), "name": name})
            if depth >= max_depth or name in SKIP_NAMES:
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), depth + 1))
            except OSError:
                continue
    return sorted(found, key=lambda item: item["path"])


def binding(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "present": path.is_file(), "SHA256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--phase21-root", type=Path, default=Path("/home/liuyeqiang/WWW_jev_phase21"))
    parser.add_argument("--scan-root", action="append", type=Path, default=None)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.resolve()
    phase21 = args.phase21_root.resolve()
    roots = [p.resolve() for p in (args.scan_root or [Path("/data1/liuyeqiang"), Path("/home/liuyeqiang")])]
    phase20 = phase21 / "reports" / "JEV_PHASE20"
    proposal = phase21 / "docs" / "PHASE20_NEXT_DATA_SCOPE_PROPOSAL.md"
    feasibility = phase20 / "NEW_DATA_SOURCE_FEASIBILITY.json"
    final = phase20 / "FINAL_GO_NO_GO.json"
    try:
        feasibility_json = json.loads(feasibility.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - explicit audit failure record
        feasibility_json = {"read_error": repr(exc)}
    try:
        final_json = json.loads(final.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        final_json = {"read_error": repr(exc)}

    matches: List[Dict[str, Any]] = []
    for root in roots:
        matches.extend(scan_candidates(root, args.max_depth))
    # The root repositories themselves are not dataset authorization. Mark only
    # actual candidate directories/files and leave unrelated names auditable.
    candidate_matches = [m for m in matches if m.get("is_dir")]
    result: Dict[str, Any] = {
        "status": "NO_SOURCE_PATH_ESTABLISHED_A_STAGE_NOT_EXECUTED",
        "captured_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase XX A-stage local-path and official-documentation audit only",
        "downloads": {"images": False, "labels": False, "videos": False, "models": False},
        "training_or_native_branches": False,
        "new_retained_bytes": 0,
        "candidate_sources": ["MMPTRACK", "WILDTRACK", "SCOUT"],
        "scan": {
            "roots": [str(root) for root in roots],
            "max_depth": args.max_depth,
            "candidate_name_matches": candidate_matches,
            "candidate_local_copy_detected": bool(candidate_matches),
            "candidate_path_authorization": "UNRESOLVED" if candidate_matches else "NOT_FOUND",
            "authorized_dataset_path_established": False,
            "interpretation": (
                "A matching local directory name was found, but it is only a candidate copy: "
                "no provenance or user-provided access approval was established."
                if candidate_matches
                else "No matching local directory was found."
            ),
        },
        "phase20_binding": {
            "proposal": binding(proposal),
            "feasibility": binding(feasibility),
            "final_go_no_go": binding(final),
            "feasibility_status": feasibility_json.get("status"),
            "no_new_dataset_path_established": feasibility_json.get("no_new_dataset_path_established"),
            "phase20_next_required": final_json.get("next_required"),
        },
        "gate": {
            "A_stage": "NOT_EXECUTED_UNTIL_AUTHORIZED_SOURCE_PATH_EXISTS",
            "B_stage": "NOT_AUTHORIZED",
            "independent_training": "NOT_AUTHORIZED",
            "official_TEST": False,
            "Full24": False,
            "sealed20_21_22": True,
            "prior_8TRAIN_3RES_gate_unchanged": True,
        },
        "required_user_action": "Provide a legally approved local copy and its directory path, or complete the official access process; this agent will not sign terms, send requests, bypass access controls, or download the dataset.",
        "next_outputs_after_authorized_path": [
            "SOURCE_ACCESS_AUDIT.json",
            "LABEL_VALIDITY_AUDIT.json",
            "PROSPECTIVE_SPLIT.json",
            "RESOURCE_MANIFEST.json",
        ],
    }

    out_json = (args.output_json or (repo / "reports" / "JEV_RECOVERY" / "NEXT_SCOPE_ACCESS_AUDIT_20261014.json")).resolve()
    out_md = (args.output_md or (repo / "docs" / "JEV_RECOVERY" / "NEXT_SCOPE_ACCESS_AUDIT_20261014.md")).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Next independent-data access audit (2026-10-14)",
        "",
        "This is a read-only A-stage audit. No candidate dataset image, label, video, model, or native branch was read or downloaded.",
        "",
        f"- Status: `{result['status']}`",
        f"- Candidate local directories matching MMPTRACK/SCOUT/WILDTRACK: `{len(candidate_matches)}`",
        f"- Candidate local copy detected: `{str(bool(candidate_matches)).lower()}`; authorization: `{'UNRESOLVED' if candidate_matches else 'NOT_FOUND'}`",
        "- Authorized dataset path established: `false`",
        "- New retained bytes: `0`",
        "- B-stage training/native intervention: `NOT_AUTHORIZED`",
        "",
        "The official feasibility record identifies MMPTRACK, WILDTRACK, and SCOUT as candidates only. A historical local directory name was found, but its authorization and provenance remain unresolved; a candidate name or public webpage is not an authorized copy. The next step requires a user-provided approved path or completion of the official access process; this audit does not sign terms or bypass access controls.",
        "",
        "The machine-readable report binds the scope proposal and feasibility/NO-GO records by SHA-256 and records the exact scan roots.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "matches": len(candidate_matches), "retained_bytes": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
