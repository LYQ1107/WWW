"""Check that view-aware JEV trace events address frozen-cache records."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Set, Tuple

from gtr.modeling.jev_perception_cache import FrozenPerceptionCache


DECISION_QUESTIONS = {
    "MATCH_DECISION",
    "MEMORY_DECISION",
    "REACTIVATION_DECISION",
}


def validate(trace: Path, cache_root: Path) -> Mapping[str, object]:
    cache = FrozenPerceptionCache(cache_root)
    cache_keys: Set[Tuple[int, int, int]] = set(cache.keys())
    events = 0
    decision_events = 0
    missing_view = 0
    missing_cache_key = 0
    malformed = 0
    questions = Counter()
    views = Counter()
    examples = []
    with trace.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            events += 1
            try:
                event = json.loads(line)
                question = str(event.get("question"))
                questions[question] += 1
                if question not in DECISION_QUESTIONS:
                    continue
                decision_events += 1
                context = event.get("context") or {}
                video = int(context["video_id"])
                frame = int(context["frame"])
                view_value = context.get("view")
                if view_value is None:
                    missing_view += 1
                    if len(examples) < 20:
                        examples.append({"line": line_number, "error": "missing_view"})
                    continue
                view = int(view_value)
                views[view] += 1
                key = (video, frame, view)
                if key not in cache_keys:
                    missing_cache_key += 1
                    if len(examples) < 20:
                        examples.append({"line": line_number, "error": "missing_cache_key", "key": list(key)})
            except Exception as exc:  # noqa: BLE001 - report malformed lines
                malformed += 1
                if len(examples) < 20:
                    examples.append({"line": line_number, "error": repr(exc)})
    aligned = not missing_view and not missing_cache_key and not malformed
    return {
        "status": "PASS" if aligned else "FAIL",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trace": str(trace),
        "cache": str(cache_root),
        "cache_records": len(cache_keys),
        "events": events,
        "decision_events": decision_events,
        "missing_view": missing_view,
        "missing_cache_key": missing_cache_key,
        "malformed_events": malformed,
        "views": dict(sorted((str(key), value) for key, value in views.items())),
        "questions": dict(sorted(questions.items())),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.trace.resolve(), args.cache.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
