"""NAS anchors and verified output migration, independent of model execution."""
from contextlib import contextmanager
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import shutil
import time
import json
from datetime import datetime, timezone

from run_worldobject_v5_controller import write_json

NAS_MOUNT = Path("/data1/liuyeqiang/nas/A100")


def mount_record(mount=NAS_MOUNT):
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, source = left.split(), right.split()
        if fields[4] == str(mount):
            return {"mount_id": int(fields[0]), "mountpoint": fields[4],
                    "filesystem": source[0], "source": source[1]}
    return None


def fd_mount_id(fd):
    for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines():
        if line.startswith("mnt_id:"):
            return int(line.split(":", 1)[1])
    raise OSError(errno.ESTALE, "FILE_DESCRIPTOR_MOUNT_ID_MISSING")


@contextmanager
def pinned_nas_root(root, mount=NAS_MOUNT):
    """Hold the actual mounted filesystem, including across lazy unmount.

    All remote writes resolve through the held directory FD. They can fail on
    disconnect, but cannot fall through into the local directory below a mount.
    """
    root, mount = Path(root), Path(mount)
    relative = root.relative_to(mount)
    expected = mount_record(mount)
    if not expected or expected["filesystem"] != "fuse.sshfs":
        raise OSError(errno.ENOTCONN, "VERIFIED_NAS_MOUNT_REQUIRED")
    mount_fd = os.open(mount, os.O_RDONLY | os.O_DIRECTORY)
    root_fd = None
    try:
        if fd_mount_id(mount_fd) != expected["mount_id"]:
            raise OSError(errno.ESTALE, "NAS_CHANGED_BEFORE_DIRECTORY_ANCHOR")
        root_fd = os.open(str(relative), os.O_RDONLY | os.O_DIRECTORY, dir_fd=mount_fd)
        if fd_mount_id(root_fd) != expected["mount_id"]:
            raise OSError(errno.ESTALE, "NAS_ROOT_IS_NOT_ON_EXPECTED_MOUNT")
        yield Path(f"/proc/self/fd/{root_fd}"), expected
        if mount_record(mount) != expected:
            raise OSError(errno.ESTALE, "NAS_CHANGED_DURING_COPY_READBACK")
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(mount_fd)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verified_copy(source, destination, expected_sha, resume_state=None, copy_identity=None):
    """Publish a file once; preserve an existing different file on collision."""
    if resume_state is not None:
        return resumable_verified_copy(source, destination, expected_sha, resume_state, copy_identity)
    source, destination = Path(source), Path(destination)
    if digest(source) != expected_sha:
        raise ValueError("LOCAL_OUTPUT_CHANGED_BEFORE_ARCHIVE")
    if destination.exists():
        if digest(destination) != expected_sha:
            raise ValueError("EXISTING_NAS_OUTPUT_DIFFERS_PRESERVE_BOTH")
        return {"mode": "FULL_REMOTE_SHA256", "verified_bytes": source.stat().st_size, "remote_sha256": expected_sha}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial." + str(os.getpid()))
    if temporary.exists():
        if digest(temporary) == expected_sha:
            os.link(temporary, destination)
            temporary.unlink()
            return {"mode": "FULL_REMOTE_SHA256", "verified_bytes": source.stat().st_size, "remote_sha256": expected_sha}
        temporary.rename(temporary.with_name(temporary.name + ".failed." + str(time.time_ns())))
    with source.open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if digest(temporary) != expected_sha:
        raise ValueError("NAS_OUTPUT_READBACK_SHA_MISMATCH")
    os.link(temporary, destination)
    temporary.unlink()
    return {"mode": "FULL_REMOTE_SHA256", "verified_bytes": source.stat().st_size, "remote_sha256": expected_sha}


def resumable_verified_copy(source, destination, expected_sha, state_path, copy_identity=None, chunk_bytes=1024 * 1024):
    """Append immutable source bytes with persistent per-chunk remote readback.

    Only acknowledged prefixes survive a retry; an unverified tail is retried.
    Each source byte is compared with remote bytes before its range is committed.
    The complete contiguous range proof therefore establishes byte equality
    without requiring one uninterrupted large remote checksum pass. The final
    native/adoption verifiers still independently check the whole output SHA.
    """
    source, destination, state_path = Path(source), Path(destination), Path(state_path)
    if digest(source) != expected_sha:
        raise ValueError("LOCAL_OUTPUT_CHANGED_BEFORE_ARCHIVE")
    size, identity = source.stat().st_size, copy_identity or str(destination)
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema": "v5_append_only_remote_readback_v1", "source_sha256": expected_sha,
        "source_bytes": size, "copy_identity": identity, "committed_bytes": 0, "chunks": []}
    if (state["source_sha256"] != expected_sha or state["source_bytes"] != size
            or state["copy_identity"] != identity):
        raise ValueError("COPY_RESUME_SOURCE_OR_DESTINATION_CHANGED")
    # Check the contiguous proof against the immutable local source. Earlier
    # remote ranges are append-only and were read back before local commit.
    offset = 0
    with source.open("rb") as reader:
        for chunk in state["chunks"]:
            if chunk["offset"] != offset or chunk["bytes"] <= 0:
                raise ValueError("NONCONTIGUOUS_COPY_READBACK_PROOF")
            payload = reader.read(chunk["bytes"])
            if len(payload) != chunk["bytes"] or hashlib.sha256(payload).hexdigest() != chunk["sha256"]:
                raise ValueError("LOCAL_SOURCE_DIFFERS_FROM_COMMITTED_COPY_RANGES")
            offset += len(payload)
    if offset != state["committed_bytes"] or offset > size:
        raise ValueError("INVALID_COMMITTED_REMOTE_PREFIX")
    temporary = destination.with_name(destination.name + ".verified-upload." + expected_sha)

    def check_last_chunk(path):
        if path.stat().st_size < offset:
            raise ValueError("REMOTE_COMMITTED_PREFIX_WAS_TRUNCATED_KEEP_LOCAL")
        if state["chunks"]:
            last = state["chunks"][-1]
            with path.open("rb") as reader:
                reader.seek(last["offset"])
                data = reader.read(last["bytes"])
            if hashlib.sha256(data).hexdigest() != last["sha256"]:
                raise ValueError("REMOTE_COMMITTED_PREFIX_CHANGED_KEEP_LOCAL")

    def finish(path):
        path.chmod(0o444)
        state.update({"complete": True, "remote_mtime_ns": path.stat().st_mtime_ns,
                      "completed_UTC": datetime.now(timezone.utc).isoformat()})
        write_json(state_path, state)
        proof_bytes = state_path.read_bytes()
        proof_sha = hashlib.sha256(proof_bytes).hexdigest()
        proof_path = state_path.with_name(state_path.name + ".complete." + proof_sha + ".json")
        if proof_path.exists():
            if digest(proof_path) != proof_sha:
                raise ValueError("IMMUTABLE_COPY_PROOF_CHANGED_KEEP_LOCAL")
        else:
            with proof_path.open("xb") as stream:
                stream.write(proof_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        return {"mode": "APPEND_ONLY_PER_CHUNK_REMOTE_BYTE_COMPARISON", "verified_bytes": size,
                "chunk_count": len(state["chunks"]), "expected_whole_file_sha256": expected_sha,
                "contiguous_remote_range_proof_sha256": proof_sha,
                "immutable_contiguous_remote_range_proof": str(proof_path),
                "final_native_or_adoption_full_SHA_still_required": True}

    if destination.exists():
        if state.get("complete") and offset == size and destination.stat().st_size == size:
            if destination.stat().st_mtime_ns != state["remote_mtime_ns"]:
                raise ValueError("IMMUTABLE_REMOTE_ARCHIVE_MODIFIED_KEEP_LOCAL")
            check_last_chunk(destination)
            return finish(destination)
        if digest(destination) != expected_sha:
            raise ValueError("EXISTING_NAS_OUTPUT_DIFFERS_PRESERVE_BOTH")
        return {"mode": "FULL_REMOTE_SHA256", "verified_bytes": size, "remote_sha256": expected_sha}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if offset and not temporary.exists():
        raise ValueError("ACKNOWLEDGED_REMOTE_PREFIX_MISSING_KEEP_LOCAL")
    if temporary.exists():
        check_last_chunk(temporary)
    with source.open("rb") as reader, temporary.open("r+b" if temporary.exists() else "x+b") as writer:
        # A failed write may have left an unacknowledged tail. The verified
        # prefix is never rewritten; only that own transfer tail is retried.
        writer.truncate(offset)
        reader.seek(offset)
        while offset < size:
            data = reader.read(min(chunk_bytes, size - offset))
            if not data:
                raise ValueError("LOCAL_OUTPUT_TRUNCATED_DURING_COPY")
            writer.seek(offset)
            if writer.write(data) != len(data):
                raise OSError(errno.EIO, "SHORT_REMOTE_CHUNK_WRITE")
            writer.flush()
            os.fsync(writer.fileno())
            writer.seek(offset)
            if writer.read(len(data)) != data:
                raise ValueError("REMOTE_CHUNK_READBACK_DIFFERS_KEEP_LOCAL")
            state["chunks"].append({"offset": offset, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "remote_readback_UTC": datetime.now(timezone.utc).isoformat()})
            offset += len(data)
            state["committed_bytes"] = offset
            write_json(state_path, state)
        writer.truncate(size)
        writer.flush()
        os.fsync(writer.fileno())
    if digest(source) != expected_sha:
        raise ValueError("LOCAL_OUTPUT_CHANGED_DURING_ARCHIVE")
    os.link(temporary, destination)
    temporary.unlink()
    return finish(destination)


def replace_local_with_verified_symlink(folder, destination):
    """Atomic directory/symlink exchange; no missing-path interval for readers.

    Call only after full remote byte verification and after its worker exits.
    The original directory remains available under the exchanged name until
    cleanup; existing open file handles also remain valid.
    """
    folder = Path(folder)
    if folder.is_symlink():
        if os.readlink(folder) != str(destination):
            raise ValueError("EXISTING_PREDICTION_LINK_DIFFERS")
        return
    holding = folder.with_name(folder.name + ".verified-migration." + str(os.getpid()))
    holding.symlink_to(destination, target_is_directory=True)
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(folder), -100, os.fsencode(holding), 2)
    except AttributeError:
        # This host has an older libc wrapper but an x86_64 Linux kernel with
        # renameat2 (__NR_renameat2=316 in its installed asm/unistd_64.h).
        if os.uname().machine != "x86_64":
            holding.unlink()
            raise OSError(errno.ENOSYS, "ATOMIC_DIRECTORY_EXCHANGE_REQUIRED_LOCAL_FILES_PRESERVED")
        libc.syscall.restype = ctypes.c_long
        result = libc.syscall(ctypes.c_long(316), ctypes.c_int(-100), ctypes.c_char_p(os.fsencode(folder)),
                              ctypes.c_int(-100), ctypes.c_char_p(os.fsencode(holding)), ctypes.c_uint(2))
    if result:
        code = ctypes.get_errno()
        holding.unlink()
        raise OSError(code, "ATOMIC_DIRECTORY_EXCHANGE_FAILED_LOCAL_FILES_PRESERVED")
    shutil.rmtree(holding)
