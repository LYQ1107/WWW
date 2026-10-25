#!/usr/bin/env python3
"""Read-only checkpoint observer; never signals or edits a training process."""
import argparse
import ctypes
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import select
import struct
import threading
import time
import torch

METHODS = ("FULL_WORLD_OBJECT_V3", "HEAD_ONLY_NO_GEOMETRY_V3", "FULL_NO_GEOMETRY_V3")
SOURCE = "6bcdcdb1a6b3683c4e406b0791416a77e5e9e64f165fbad4a61667698b64645f"


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    output = root / "audit_snapshots"
    output.mkdir(parents=True, exist_ok=True)
    lock_file = (output / "observer.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index_path = output / "epoch_archive_index.json"
    index = read(index_path) if index_path.exists() else {"source_sha256": SOURCE, "records": []}
    mutex = threading.Lock()
    stop = threading.Event()
    errors = []
    event_support = {}

    def capture(method, trigger):
        directory = root / "train" / method
        job = read(root / "jobs" / ("train_" + method + ".status.json"))
        if job["source_sha256"] != SOURCE:
            raise ValueError("TRAIN_SOURCE_CHANGED")
        # Open the atomic complete checkpoint immediately. The open inode stays
        # readable even if the trainer replaces resume.pt again afterwards.
        with (directory / "resume.pt").open("rb") as stream:
            payload = stream.read()
        digest = hashlib.sha256(payload).hexdigest()
        with mutex:
            if any(row["method"] == method and row["sha256"] == digest for row in index["records"]):
                return
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if state["stage"] not in ("selection", "final"):
            raise ValueError("UNKNOWN_SAFE_BOUNDARY_STAGE")
        destination = output / method / (digest + ".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open("xb") as stream:
                stream.write(payload)
        elif hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError("EXISTING_ARCHIVE_SHA_MISMATCH")
        try:
            meta = read(directory / "resume.json")
            bound = meta["sha256"] == digest and meta["source_sha256"] == SOURCE
        except (OSError, json.JSONDecodeError):
            bound = False
        row = {"method": method, "path": str(destination), "sha256": digest,
               "stage": state["stage"], "selection_epochs": len(state["selection"]),
               "final_epochs": len(state["final_history"]), "selected_final_epochs": state["selected"],
               "usable_as_selection_epoch_checkpoint": state["stage"] == "selection",
               "resume_metadata_SHA_matched_at_capture": bound,
               "source_binding": "source-pinned training job plus atomically replaced complete checkpoint",
               "training_worker_pid": job["worker_pid"], "observed_at": time.time(), "trigger": trigger}
        with mutex:
            index["records"].append(row)
            atomic(index_path, index)
        print(json.dumps({"checkpoint_archived": row}), flush=True)

    def watch(method):
        descriptor = None
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.inotify_init1.argtypes = [ctypes.c_int]
            libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            descriptor = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if descriptor < 0 or libc.inotify_add_watch(descriptor,
                    os.fsencode(root / "train" / method), 0x80) < 0:
                raise OSError(ctypes.get_errno(), "inotify watch unavailable")
            event_support[method] = True
        except (AttributeError, OSError):
            if descriptor is not None and descriptor >= 0:
                os.close(descriptor)
            descriptor = None
            event_support[method] = False
        signature = None
        while not stop.is_set():
            try:
                event = False
                if descriptor is not None and select.select([descriptor], [], [], .2)[0]:
                    data = os.read(descriptor, 65536)
                    offset = 0
                    while offset < len(data):
                        _, _, _, length = struct.unpack_from("iIII", data, offset)
                        name = data[offset + 16:offset + 16 + length].split(b"\0", 1)[0]
                        offset += 16 + length
                        event = event or name == b"resume.pt"
                elif descriptor is None:
                    stop.wait(.2)
                path = root / "train" / method / "resume.pt"
                stat = path.stat()
                current = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                if event or current != signature:
                    capture(method, "ATOMIC_RENAME_EVENT" if event else "INITIAL_OR_POLL")
                    signature = current
            except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
                with mutex:
                    errors.append({"time": time.time(), "method": method, "error": str(exc)})
                    del errors[:-20]
                stop.wait(.5)
        if descriptor is not None:
            os.close(descriptor)

    threads = [threading.Thread(target=watch, args=(method,), daemon=True) for method in METHODS]
    for thread in threads:
        thread.start()
    try:
        while not (output / "observer.stop").exists():
            with mutex:
                atomic(output / "observer_status.json", {"status": "READ_ONLY_EPOCH_OBSERVER_RUNNING",
                    "pid": os.getpid(), "updated_at": time.time(), "source_sha256": SOURCE,
                    "inotify_enabled": dict(event_support), "CUDA_context_initialized": torch.cuda.is_initialized(),
                    "archived_checkpoints": len(index["records"]), "recent_observation_errors": list(errors),
                    "training_signals_sent": 0, "training_files_written": 0})
            complete = all((root / "train" / method / "train_gate.json").is_file() for method in METHODS)
            if complete:
                for method in METHODS:
                    capture(method, "FINAL_GATE_PRESENT")
                break
            time.sleep(5)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
    print(json.dumps({"epoch_observer_finished": True}), flush=True)


if __name__ == "__main__":
    main()
