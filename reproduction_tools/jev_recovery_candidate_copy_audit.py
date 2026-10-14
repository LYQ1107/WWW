"""Audit a discovered candidate dataset copy without consuming its data.

The Phase XX proposal permits a small metadata-only A-stage audit.  This tool
reads only the existing audit JSON files in a user-provided candidate root and
uses filesystem metadata for size/count information.  It never opens image,
label, video, model, or archive payload members and never trains or evaluates
a native tracker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_ROOT = Path(
    "/data1/liuyeqiang/WWW_jev_phase14_runtime/20261010_v1/"
    "external_wildtrack_v1"
)
METADATA_FILES = (
    "PROGRESS.json",
    "RESULT_METADATA.json",
    "RESULT.json",
    "ARCHIVE_DIRECTORY.json",
    "ZIP_OFFSET_COMPATIBILITY.json",
)
PROVENANCE_TOKENS = (
    "license",
    "readme",
    "terms",
    "consent",
    "provenance",
    "authorization",
    "authorisation",
)


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def walk_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not Path(current, d).is_symlink())
        for name in sorted(files):
            path = Path(current, name)
            if not path.is_symlink():
                yield path


def payload_inventory(root: Path) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    bytes_by_suffix: Dict[str, int] = {}
    total_bytes = 0
    total_files = 0
    for path in walk_files(root):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        suffix = path.suffix.lower() or "<no_extension>"
        counts[suffix] = counts.get(suffix, 0) + 1
        bytes_by_suffix[suffix] = bytes_by_suffix.get(suffix, 0) + size
        total_bytes += size
        total_files += 1
    return {
        "file_count": total_files,
        "disk_bytes_from_file_stat": total_bytes,
        "counts_by_suffix": dict(sorted(counts.items())),
        "bytes_by_suffix": dict(sorted(bytes_by_suffix.items())),
    }


def compact_result(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not value:
        return {"present": False}
    binding = value.get("binding")
    gt = value.get("GT_metadata")
    return {
        "present": True,
        "status": value.get("status"),
        "archive_bytes": value.get("archive_bytes"),
        "network_bytes": value.get("network_bytes"),
        "total_downloaded_members": value.get("total_downloaded_members"),
        "HTTP_range_requests": value.get("HTTP_range_requests"),
        "GT_training_used": value.get("GT_training_used"),
        "tracking_results_viewed": value.get("tracking_results_viewed"),
        "all_other_views_and_later_frames_unread": value.get(
            "all_other_views_and_later_frames_unread"
        ),
        "protocol_SHA256": value.get("protocol_SHA256"),
        "binding": {
            key: binding.get(key)
            for key in (
                "TRAIN",
                "development",
                "heldout",
                "official_TEST",
                "Full24",
                "source_commit",
                "frozen_evidence_SHA256",
                "preregistration_SHA256",
            )
        }
        if isinstance(binding, dict)
        else None,
        "GT_metadata": {
            key: gt.get(key)
            for key in (
                "frames",
                "distinct_personID",
                "IDs_repeated_across_frames",
                "person_count_min_max",
            )
        }
        if isinstance(gt, dict)
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    root = args.candidate_root.resolve()
    metadata: Dict[str, Any] = {}
    metadata_bytes = 0
    metadata_bindings: Dict[str, Any] = {}
    for name in METADATA_FILES:
        path = root / name
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            size = 0
        metadata_bytes += size
        metadata_bindings[name] = {
            "path": str(path),
            "present": path.is_file(),
            "bytes": size,
            "SHA256": sha256(path),
        }
        parsed = read_json(path)
        if parsed is not None:
            metadata[name] = parsed

    top_level_names = []
    if root.is_dir():
        try:
            top_level_names = sorted(entry.name for entry in root.iterdir())
        except OSError:
            top_level_names = []
    provenance_names = [
        name
        for name in top_level_names
        if any(token in name.lower() for token in PROVENANCE_TOKENS)
    ]
    inventory = payload_inventory(root)
    result_metadata = metadata.get("RESULT_METADATA.json")
    result = metadata.get("RESULT.json")
    selected = result or result_metadata or {}
    result = {
        "status": "CANDIDATE_COPY_FOUND_AUTHORIZATION_UNRESOLVED",
        "captured_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase XX A-stage metadata and filesystem-stat audit only",
        "candidate_root": str(root),
        "candidate_local_copy_detected": root.is_dir(),
        "authorized_dataset_path_established": False,
        "authorization_status": "UNRESOLVED",
        "authorization_evidence_found_in_candidate_root": bool(provenance_names),
        "provenance_or_access_files_named_at_root": provenance_names,
        "metadata_budget": {
            "metadata_bytes_read": metadata_bytes,
            "limit_bytes": 64 * 1024 * 1024,
            "within_limit": metadata_bytes <= 64 * 1024 * 1024,
        },
        "metadata_files": metadata_bindings,
        "metadata_summary": {
            "RESULT_METADATA.json": compact_result(result_metadata),
            "RESULT.json": compact_result(result),
        },
        "filesystem_inventory": inventory,
        "reads_or_actions_not_performed": {
            "image_payloads": True,
            "label_payloads": True,
            "video_payloads": True,
            "model_payloads": True,
            "archive_payloads": True,
            "training": True,
            "native_tracking_or_intervention": True,
            "official_TEST": True,
            "Full24": True,
        },
        "gate": {
            "A_stage": "METADATA_ONLY_CANDIDATE_COPY_AUDIT",
            "B_stage": "NOT_AUTHORIZED",
            "independent_training": "NOT_AUTHORIZED",
            "prior_8TRAIN_3RES_gate_unchanged": True,
        },
        "interpretation": (
            "This is an existing historical Wildtrack-named copy, not evidence of "
            "legal access. The local metadata has no named authorization/provenance "
            "file; a user-provided approval or official access record is still required."
        ),
        "required_user_action": (
            "Confirm a legally approved path and its permitted TRAIN/VAL scope, or "
            "complete the official access process. Do not use this copy for training "
            "or native evaluation until that confirmation is recorded."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    inventory_text = ", ".join(
        f"{key}={value}" for key, value in inventory["counts_by_suffix"].items()
    )
    lines = [
        "# Historical candidate-copy audit (2026-10-14)",
        "",
        "A metadata-only audit found a historical `Wildtrack_dataset` copy. It did not open image, label, video, model, or archive payloads and did not train or run native tracking.",
        "",
        f"- Candidate root: `{root}`",
        f"- Status: `{result['status']}`",
        f"- Existing files by suffix: `{inventory_text}`",
        f"- Existing filesystem bytes (stat sum): `{inventory['disk_bytes_from_file_stat']}`",
        f"- Metadata bytes read: `{metadata_bytes}` (limit 67108864)",
        "- Authorized dataset path: `false`",
        "- Authorization/provenance status: `UNRESOLVED`",
        "- New retained bytes: `0`",
        "- B-stage training/native intervention: `NOT_AUTHORIZED`",
        "",
        "The local files contain a previous audit binding and a completed member download state, but those facts do not establish that the current user or project may use the copy. The official access record must be supplied before any image/label use or independent experiment.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_root": str(root),
                "metadata_bytes_read": metadata_bytes,
                "existing_disk_bytes": inventory["disk_bytes_from_file_stat"],
                "new_retained_bytes": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
