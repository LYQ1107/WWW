"""Freeze a metadata-only audit of independent dataset source candidates.

The audit reads small, already captured official-page snapshots and access
records.  It does not contact a host, download a dataset, open image/label
payloads, or send a registration or license request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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


def file_record(path: Path, **extra: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "SHA256": sha256(path),
    }
    record.update(extra)
    return record


def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("/home/liuyeqiang/WWW_jev_master_audit_runtime"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    audit_root = args.audit_root.resolve()
    source_root = audit_root / "public_source_review_v1"
    alternative_root = audit_root / "alternative_sources_20261012"

    wildtrack_html = source_root / "wildtrack_release.html"
    wildtrack_head = source_root / "wildtrack_archive_head.json"
    mmp_html = source_root / "mmp_release.html"
    scout_readme = source_root / "cvlab-epfl" / "scout_toolkit" / "README.md"
    scout_meta = source_root / "scout_metadata.json"
    divo_readme = alternative_root / "shengyuhao_DIVOTrack_README.md"
    baidu_check = alternative_root / "BAIDU_SHARE_PAGE_CHECK.json"
    candidate_report = repo / "reports" / "JEV_RECOVERY" / "NEXT_SCOPE_CANDIDATE_COPY_AUDIT_20261014.json"
    try:
        candidate = json.loads(candidate_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        candidate = {}

    wildtrack_text = text(wildtrack_html).lower()
    mmp_text = text(mmp_html).lower()
    scout_text = text(scout_readme).lower()
    divo_text = text(divo_readme).lower()
    try:
        wildtrack_head_json = json.loads(wildtrack_head.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        wildtrack_head_json = {}
    try:
        baidu_json = json.loads(baidu_check.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        baidu_json = []

    sources = {
        "WILDTRACK": {
            "scope_role": "primary_candidate",
            "official_url": "https://www.epfl.ch/labs/cvlab/data/data-wildtrack/",
            "access_status": "PUBLIC_DOWNLOAD_LICENSE_UNRESOLVED",
            "facts": {
                "official_page_lists_annotated_archive": "wildtrack_dataset_full.zip" in wildtrack_text,
                "archive_head_status": wildtrack_head_json.get("status"),
                "archive_bytes": wildtrack_head_json.get("headers", {}).get("Content-Length"),
                "official_page_explicit_license_or_terms_token": bool(
                    re.search(r"license|licence|terms of use|terms and conditions", wildtrack_text)
                ),
                "annotation_rate_and_frames": "400 frames at 2fps" if "400 frames" in wildtrack_text else None,
            },
            "evidence": [
                file_record(wildtrack_html),
                file_record(wildtrack_head),
            ],
            "decision": "Do not treat the historical local copy as authorized until provenance and permitted research use are confirmed.",
        },
        "MMPTRACK": {
            "scope_role": "primary_candidate",
            "official_url": "https://iccv2021-mmp.github.io/subpage/dataset.html",
            "access_status": "SIGN_TERMS_AND_EMAIL_REQUIRED",
            "facts": {
                "requires_signed_terms": "sign" in mmp_text and "terms and conditions" in mmp_text,
                "requires_email_return": "email back" in mmp_text,
                "research_only": "research purpose" in mmp_text,
            },
            "evidence": [file_record(mmp_html)],
            "decision": "No access request or signed agreement is present; do not download or train.",
        },
        "SCOUT": {
            "scope_role": "primary_candidate",
            "official_url": "https://scout.epfl.ch/",
            "access_status": "OFFICIAL_SITE_TERMS_REQUIRED_ACCESS_UNRESOLVED",
            "facts": {
                "toolkit_defers_to_official_terms": "official scout dataset website" in scout_text and "licensing information" in scout_text,
                "metadata_snapshot_present": scout_meta.is_file(),
            },
            "evidence": [file_record(scout_readme), file_record(scout_meta)],
            "decision": "The toolkit is not an authorization record; no approved local copy or completed registration is recorded.",
        },
        "DIVOTrack": {
            "scope_role": "secondary_candidate_not_admitted_to_current_three_source_scope",
            "official_url": "https://github.com/shengyuhao/DIVOTrack",
            "access_status": "AUTHORITY_AND_LICENSE_SUBMISSION_REQUIRED",
            "facts": {
                "download_hosted_on_huggingface": "huggingface" in divo_text,
                "license_submission_before_unzip": "sending us" in divo_text and "license" in divo_text,
                "authority_or_license_agreement": "license agreement" in divo_text or "apply for authority" in divo_text,
            },
            "evidence": [file_record(divo_readme)],
            "decision": "Documented as a secondary option only; it is not added to the three-source preregistered scope.",
        },
    }
    baidu_records = baidu_json if isinstance(baidu_json, list) else []
    result: Dict[str, Any] = {
        "status": "NO_AUTHORIZED_SOURCE_B_STAGE_REMAINS_CLOSED",
        "captured_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase XX source-candidate documentation and local provenance audit only",
        "primary_candidate_count": 3,
        "sources": sources,
        "baidu_share_checks": baidu_records,
        "historical_local_copy": {
            "candidate_report": str(candidate_report),
            "path": candidate.get("candidate_root"),
            "authorization_status": candidate.get("authorization_status"),
            "authorized_dataset_path_established": candidate.get("authorized_dataset_path_established"),
        },
        "actions_not_performed": {
            "network_download": True,
            "image_or_label_payload_read": True,
            "registration_or_license_request": True,
            "training": True,
            "native_evaluation": True,
        },
        "gate": {
            "A_stage": "SOURCE_DOCUMENTATION_AND_LOCAL_PROVENANCE_AUDIT_COMPLETE",
            "B_stage": "NOT_AUTHORIZED",
            "independent_training": "NOT_AUTHORIZED",
            "official_TEST": False,
            "Full24": False,
            "sealed20_21_22": True,
        },
        "interpretation": (
            "A public download link, toolkit, GitHub repository, or Baidu share page "
            "is not by itself proof that this project has permission to use the data. "
            "The three-source scope remains frozen until an approved path and permitted "
            "TRAIN/VAL use are recorded."
        ),
        "required_user_action": (
            "Provide a legally approved local path and its permitted scope, or complete "
            "the relevant official access process."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Independent source-candidate audit (2026-10-14)",
        "",
        "This is a documentation and local-provenance audit. It did not download data, open image/label payloads, send registration/license requests, train, or run native evaluation.",
        "",
        f"- Status: `{result['status']}`",
        "- Primary candidate scope: `WILDTRACK`, `MMPTRACK`, `SCOUT`",
        "- Secondary candidate reviewed but not admitted: `DIVOTrack`",
        f"- Historical local copy authorization: `{candidate.get('authorization_status')}`",
        "- B-stage training/native intervention: `NOT_AUTHORIZED`",
        "",
        "WILDTRACK has an official download page and a live archive head, but the captured official page does not state a clear project-use license. MMPTRACK explicitly requires signed terms returned by email. SCOUT defers licensing to its official site. DIVOTrack requires authority/license submission before unzipping. Baidu share pages were checked only for page availability and did not expose a verified file list or download authorization.",
        "",
        "A public URL is therefore recorded as source evidence, not as authorization. The next experiment gate remains closed until access is documented.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_candidate_count": 3,
                "historical_copy_authorization": candidate.get("authorization_status"),
                "downloads": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
