"""NAS anchors and verified output migration, independent of model execution."""
from contextlib import contextmanager
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import shutil
import time

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


def verified_copy(source, destination, expected_sha):
    """Publish a file once; preserve an existing different file on collision."""
    source, destination = Path(source), Path(destination)
    if digest(source) != expected_sha:
        raise ValueError("LOCAL_OUTPUT_CHANGED_BEFORE_ARCHIVE")
    if destination.exists():
        if digest(destination) != expected_sha:
            raise ValueError("EXISTING_NAS_OUTPUT_DIFFERS_PRESERVE_BOTH")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial." + str(os.getpid()))
    if temporary.exists():
        if digest(temporary) == expected_sha:
            os.link(temporary, destination)
            temporary.unlink()
            return
        temporary.rename(temporary.with_name(temporary.name + ".failed." + str(time.time_ns())))
    with source.open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if digest(temporary) != expected_sha:
        raise ValueError("NAS_OUTPUT_READBACK_SHA_MISMATCH")
    os.link(temporary, destination)
    temporary.unlink()


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
