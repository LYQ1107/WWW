"""Fail-closed boundary for real-data runs. No approval is shipped in this repo."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

APPROVAL_SCHEMA = "worldobject_experiment_user_approval_v3"


def source_commit():
    # Read Git refs without forking after Torch/OpenMP has initialized.
    # This also works in linked worktrees and with packed remote refs.
    return read_git_head(Path(__file__).resolve().parents[3])


def read_git_head(repository):
    entry=Path(repository)/".git"
    gitdir=entry
    if entry.is_file():
        gitdir=(entry.parent/entry.read_text().strip().split("gitdir: ",1)[1]).resolve()
    head=(gitdir/"HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref=head[5:]
    common=(gitdir/(gitdir/"commondir").read_text().strip()).resolve() if (gitdir/"commondir").is_file() else gitdir
    for directory in (gitdir,common):
        loose=directory/ref
        if loose.is_file():
            return loose.read_text().strip()
    packed=common/"packed-refs"
    for line in packed.read_text().splitlines() if packed.is_file() else []:
        if not line.startswith(("#","^")) and line.split()[-1:]==[ref]:
            return line.split()[0]
    raise RuntimeError("CANNOT_RESOLVE_SOURCE_COMMIT")


def source_fingerprint():
    root=Path(__file__).resolve().parents[2]
    files=sorted(list((root/"worldmot").rglob("*.py"))+list((root/"scripts").glob("*.py"))
                 +list((root/"tests").glob("*.py"))+list((root/"configs").glob("worldobject_v3*")))
    digest=hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root.parent)).encode()+b"\0"+path.read_bytes())
    return digest.hexdigest()


def require_experiment_authorization(path, operation, split=None, method_id=None):
    if not path:
        raise PermissionError("WAITING_USER_APPROVAL: real-data operation disabled")
    path = Path(path)
    data = json.loads(path.read_text())
    if (data.get("schema") != APPROVAL_SCHEMA or data.get("approved") is not True
            or not data.get("user_authorization_reference")
            or not data.get("source_commit")
            or data.get("source_sha256") != source_fingerprint()
            or operation not in data.get("operations", [])
            or (split is not None and split not in data.get("splits", []))
            or (method_id is not None and method_id not in data.get("method_ids", []))):
        raise PermissionError("WAITING_USER_APPROVAL: authorization scope/source mismatch")
    return {"approval_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "operation": operation, "source_commit": data["source_commit"],"execution_commit":source_commit()}
