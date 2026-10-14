"""Record a bounded screen of additional official dataset candidates.

This is a source-documentation index only.  It deliberately does not add a
candidate to the preregistered three-source scope, download data, or infer
identity supervision from a dataset name.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    screened: Dict[str, Dict[str, Any]] = {
        "EPFL_RLC": {
            "official_url": "https://www.epfl.ch/labs/cvlab/data/data-rlc/",
            "access_status": "PUBLIC_PAGE_REVIEW_ONLY_NOT_ADMITTED",
            "documented_facts": [
                "three synchronized static HD cameras",
                "4088 balanced manually annotated multi-view examples",
                "position IDs and occupied/free position labels",
                "1.1 GB dataset ZIP and 123 KB testing-sequence JSON",
            ],
            "reason_not_admitted": "The public description is a position/detection example interface, not verified person-ID lifecycle supervision for the current native controller; explicit project-use terms were not established in this screen.",
        },
        "SPORTCENTER": {
            "official_url": "https://www.epfl.ch/labs/cvlab/data/sportcenter-dataset/",
            "access_status": "RESEARCH_USE_STATED_INTERFACE_NOT_ADMITTED",
            "documented_facts": [
                "official page says the data is free for research purposes",
                "multi-view human-pose sequences from eight fixed synchronized cameras",
                "player trajectories and a subset of manually annotated poses",
            ],
            "reason_not_admitted": "The public description does not establish the required cross-camera person-ID association labels and native two-camera lifecycle interface.",
        },
        "CROSSTRACK_UAV": {
            "official_url": "https://github.com/ccs0302/CrossTrack-UAV",
            "access_status": "SIGNED_LICENSE_REQUIRED_NOT_AUTHORIZED",
            "documented_facts": [
                "official repository requests a completed and signed license agreement",
                "additional data may be available on request",
            ],
            "reason_not_admitted": "No signed agreement or approved local copy is present.",
        },
    }
    result: Dict[str, Any] = {
        "status": "NO_NEW_PRIMARY_SOURCE_ADMITTED",
        "captured_UTC": datetime.now(timezone.utc).isoformat(),
        "scope": "additional official-source landscape screen; no data access",
        "screened_sources": screened,
        "current_primary_scope_unchanged": ["WILDTRACK", "MMPTRACK", "SCOUT"],
        "new_primary_candidate_count": 0,
        "actions_not_performed": {
            "downloads": True,
            "payload_read": True,
            "registration_or_license_request": True,
            "training": True,
            "native_evaluation": True,
        },
        "decision": "Do not expand the preregistered source scope or run B-stage experiments from this screen. Require a source-specific authorization record and a native identity-label/interface audit first.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Additional official-source landscape screen (2026-10-14)",
        "",
        "This screen reads official source descriptions only. It does not download data, send access requests, or add candidates to the preregistered scope.",
        "",
        f"- Status: `{result['status']}`",
        "- New primary candidates admitted: `0`",
        "- Current primary scope unchanged: `WILDTRACK`, `MMPTRACK`, `SCOUT`",
        "",
        "| Candidate | Decision | Reason |",
        "|---|---|---|",
    ]
    for name, value in screened.items():
        lines.append(
            f"| `{name}` | `{value['access_status']}` | {value['reason_not_admitted']} |"
        )
    lines.extend(
        [
            "",
            "A dataset can be useful for a related task without satisfying the current identity-ownership and native association gate. No B-stage action is authorized by this screen.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "new_primary_candidate_count": 0}))


if __name__ == "__main__":
    main()
