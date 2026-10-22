"""Guarded entry points for the official TrackEval protocol.

This module deliberately does not map WorldMOT's custom multi-camera JSON to
MOTChallenge implicitly.  A caller must provide a reviewed converter and the
complete sequence directory; otherwise the result is marked blocked rather
than silently using ``evaluate_2d.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def locate_trackeval(search_roots: list[str | Path] | None = None) -> Path | None:
    roots = [Path.cwd()]
    here = Path(__file__).resolve()
    roots.extend([here.parents[3], here.parents[4]])
    if search_roots:
        roots.extend(Path(x) for x in search_roots)
    candidates = []
    for root in roots:
        candidates.extend([root / "TrackEval", root / "trackeval"])
    for path in candidates:
        if (path / "trackeval").is_dir() and (path / "scripts" / "run_mot_challenge.py").exists():
            return path.resolve()
    return None


def source_manifest(trackeval_root: str | Path | None = None) -> dict:
    path = Path(trackeval_root).resolve() if trackeval_root else locate_trackeval()
    if path is None:
        return {"status": "BLOCKED_TRACKEVAL_NOT_FOUND"}
    commit = None
    try:
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"status": "READY", "path": str(path), "git_commit": commit,
            "protocol": "TrackEval official metrics only"}


def require_official_source(trackeval_root: str | Path | None = None) -> dict:
    manifest = source_manifest(trackeval_root)
    if manifest["status"] != "READY":
        raise RuntimeError("official TrackEval is required; simplified WorldMOT metrics are not a fallback")
    return manifest


def evaluation_gate(*, tracker_output: str | Path, complete_sequence_root: str | Path,
                    trackeval_root: str | Path | None = None) -> dict:
    """Validate prerequisites without running an unreviewed format conversion."""
    manifest = require_official_source(trackeval_root)
    tracker_output = Path(tracker_output).resolve()
    complete_sequence_root = Path(complete_sequence_root).resolve()
    if not tracker_output.exists():
        return {**manifest, "status": "BLOCKED_TRACKER_OUTPUT_MISSING",
                "tracker_output": str(tracker_output)}
    if not complete_sequence_root.exists():
        return {**manifest, "status": "BLOCKED_COMPLETE_SEQUENCE_MISSING",
                "complete_sequence_root": str(complete_sequence_root)}
    return {**manifest, "status": "READY_FOR_REVIEWED_CONVERTER",
            "tracker_output": str(tracker_output),
            "complete_sequence_root": str(complete_sequence_root),
            "warning": "No metric is emitted until the dataset-specific converter is reviewed."}

