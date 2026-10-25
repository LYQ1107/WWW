"""V3 immutable run scope, separate from historical experiment locks."""
from __future__ import annotations
import json
from pathlib import Path
from .worldobject_cache import sha256
from ..runtime.experiment_authorization import source_fingerprint

SCOPE_SCHEMA="worldobject_frozen_scope_v3"


def load_scope_lock(path,split,method_id):
    lock=json.loads(Path(path).read_text())
    if (lock.get("schema")!=SCOPE_SCHEMA or not lock.get("source_commit")
            or lock.get("source_sha256")!=source_fingerprint()
            or lock.get("method_id")!=method_id or split not in lock.get("splits",{})):
        raise ValueError("V3_FROZEN_SCOPE_MISMATCH")
    expected=(24,28754) if split=="train" else (22,29019)
    scope=lock["splits"][split]
    if (scope.get("scene_count"),scope.get("timestamps"))!=expected:
        raise ValueError("INCOMPLETE_FROZEN_SCOPE")
    return lock


def validate_scene_lock(lock,cache,split):
    expected=lock["splits"][split]["scenes"].get(str(cache.scene))
    if not expected or expected["manifest_sha256"]!=sha256(cache.path/"manifest.json"):
        raise ValueError("SCENE_CACHE_DIFFERS_FROM_V3_FROZEN_SCOPE")
