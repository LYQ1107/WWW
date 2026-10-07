"""Observable, resumable small-video H8 build on independent GPUs.

Only output ranges are partitioned. Every branch sees the complete source
trace and future cache keys, and starts from a production-order OFF snapshot.
Successful chunks are reusable; an interrupted chunk is recomputed from its
own snapshot. This does not authorize the 24-video Full H8 build.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
BASE = Path("/home/liuyeqiang/WWW_jev_rng_v4_runtime")
CACHE = Path("/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train")
ANNOTATIONS = Path("/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json")
CHECKPOINT = Path("/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth")
BASE_COMMIT = "4108f18f5040432f68d55872e81e9f76e9acd08f"
for p in (ROOT, ROOT / "reproduction_tools", ROOT / "third_party/CenterNet2"):
    sys.path.insert(0, str(p))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text())


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               JEV_CACHE_PATH=str(CACHE),
               PYTHONPATH=":".join(str(p) for p in (ROOT, ROOT / "reproduction_tools", ROOT / "third_party/CenterNet2")))
    return env


def source_binding():
    subprocess.run(["git", "diff", "--exit-code"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return {"source_commit": commit(), "semantic_base_commit": BASE_COMMIT,
            "source_root": str(ROOT), "config": str(ROOT / "configs/VISION_test.yaml"),
            "config_sha256": digest(ROOT / "configs/VISION_test.yaml"),
            "gmt_checkpoint": str(CHECKPOINT), "gmt_checkpoint_sha256": digest(CHECKPOINT),
            "annotations": str(ANNOTATIONS), "annotations_sha256": digest(ANNOTATIONS),
            "perception_cache": str(CACHE), "perception_cache_index_sha256": digest(CACHE / "index.jsonl"),
            "horizon": 8, "association_backend": "formal_gmt_transformer",
            "counterfactual_engine": "cached_perception_mutable_association_v2",
            "state_schema_version": 2,
            "trajectory_rng_policy": "branch_local_explicit_python_random_v1",
            "trajectory_rng_master_seed": 20261006,
            "trajectory_rng_state_cloned_per_counterfactual_branch": True,
            "trajectory_rng_transformer_sha256": "sha256:" + digest(ROOT / "gtr/modeling/roi_heads/transformer.py"),
            "trajectory_rng_counterfactual_engine_sha256": "sha256:" + digest(ROOT / "reproduction_tools/jev_counterfactual_v2.py"),
            "trajectory_rng_adapter_sha256": "sha256:" + digest(ROOT / "reproduction_tools/jev_gmt_association_adapter.py"),
            "builder_sha256": digest(ROOT / "reproduction_tools/build_jev_counterfactual_v2.py"),
            "official_test_read": False, "official_test_generation_authorized": False,
            "full_h8_authorized": False}


def trace_path(video):
    return BASE / "formal_current_head_off_trace" / f"video{video:02d}" / f"trace_video_{video:02d}.jsonl"


def load_context(video, cache):
    from build_jev_counterfactual_dataset import normalize_events
    from build_jev_counterfactual_v2 import event_maps, ordered_production_keys
    events = normalize_events(trace_path(video), video_ids=[video])[video]
    actions, memories, by_key = event_maps(events)
    keys, seed = ordered_production_keys([k for k in cache.keys() if k[0] == video], lambda k: cache.load(*k))
    return events, actions, memories, by_key, keys, seed


def engine():
    from build_jev_counterfactual_v2 import build_formal_gmt_engine
    return build_formal_gmt_engine(config_file=ROOT / "configs/VISION_test.yaml",
                                  checkpoint=CHECKPOINT, device="cuda:0", view_num=2, history_limit=80)


def prepare(args):
    import torch
    from build_jev_counterfactual_v2 import advance_off_state_for_key
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    from jev_counterfactual_v2 import seed_production_state_from_payload
    from jev_intra_video_chunking import plan_chunks, save_state_snapshot
    binding = source_binding()
    if (args.output / "plan.json").exists():
        plan = read(args.output / "plan.json")
        if plan["binding"] != binding or plan["videos"] != args.videos or plan["frames"] != args.frames:
            raise RuntimeError("resume provenance or video set changed")
        return plan
    args.output.mkdir(parents=True, exist_ok=True)
    model = engine()
    cache = FrozenPerceptionCache(CACHE)
    plan = {"binding": binding, "created_utc": now(), "videos": args.videos,
            "frames": args.frames, "chunks": [], "video_metadata": {}}
    for video in args.videos:
        events, actions, memories, by_key, keys, seed = load_context(video, cache)
        start, end = 1, len(keys)
        if args.frames:
            selected = [i for i, k in enumerate(keys) if args.frames[0] <= k[1] <= args.frames[1] and i >= 1]
            if not selected:
                raise ValueError("empty requested frame range")
            start, end = min(selected), max(selected) + 1
        chunks = plan_chunks(video_id=video, ordered_keys=keys[:end], events_by_key=by_key,
                             target_records=args.chunk_records, initial_key_index=start)
        state, _ = seed_production_state_from_payload(cache.load(*seed), seed)
        cursor = 1
        for chunk in chunks:
            for i in range(cursor, chunk["key_start"]):
                k = keys[i]
                advance_off_state_for_key(payload=cache.load(*k), key=k, by_key=by_key,
                                         actions=actions, memories=memories, engine=model, state=state)
                if i % 25 == 0:
                    atomic(args.output / "progress.json", {"phase": "warmup", "updated_utc": now(),
                           "video_id": video, "key_index": i, "key": k, "pid": os.getpid()})
            name = f"video{video:02d}_chunk{chunk['chunk_index']:04d}"
            snapshot = args.output / "snapshots" / (name + ".pt")
            metadata = {"source_commit": binding["source_commit"], "video_id": video,
                        "key_start": chunk["key_start"], "trace_sha256": digest(trace_path(video)),
                        "trajectory_rng_seed": state.trajectory_rng_seed,
                        "trajectory_rng_calls": state.trajectory_rng_calls}
            snapshot_sha = save_state_snapshot(snapshot, state, metadata)
            chunk.update(name=name, snapshot=str(snapshot), snapshot_sha256=snapshot_sha,
                         status="PENDING", expected_question_counts=dict(Counter(
                             e["question"] for k in keys[chunk["key_start"]:chunk["key_end"]]
                             for e in by_key.get(k, ()) if e.get("legal_actions"))))
            plan["chunks"].append(chunk)
            cursor = chunk["key_start"]
        plan["video_metadata"][str(video)] = {"trace": str(trace_path(video)),
                    "trace_sha256": digest(trace_path(video)), "key_start": start, "key_end": end,
                    "seed_key": list(seed), "records": sum(c["decision_count"] for c in chunks)}
        print(json.dumps({"phase": "prepared", "video_id": video, "chunks": len(chunks),
                          "records": plan["video_metadata"][str(video)]["records"]}), flush=True)
    atomic(args.output / "plan.json", plan)
    atomic(args.output / "queue.json", {"binding": binding, "chunks": plan["chunks"]})
    del model, cache
    gc.collect()
    torch.cuda.empty_cache()
    return plan


def queue_update(output, fn):
    with (output / "queue.lock").open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        q = read(output / "queue.json")
        result = fn(q)
        atomic(output / "queue.json", q)
        return result


def claim(output):
    def choose(q):
        if q["binding"]["source_commit"] != commit():
            raise RuntimeError("worker source changed after snapshot preparation")
        pending = [c for c in q["chunks"] if c["status"] == "PENDING"]
        if not pending:
            return None
        c = max(pending, key=lambda c: c["decision_count"] * (1 + c["key_start"] / 500))
        c.update(status="RUNNING", pid=os.getpid(), started_utc=now(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
        return dict(c)
    return queue_update(output, choose)


def worker(args):
    from build_jev_counterfactual_dataset import load_gt
    from build_jev_counterfactual_v2 import build_v2_records
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    from jev_intra_video_chunking import load_state_snapshot
    from run_jev_full_h8_fast_worker import PayloadLRU
    plan = read(args.output / "plan.json")
    if plan["binding"] != source_binding():
        raise RuntimeError("worker inputs/source differ from preparation")
    model = engine()
    cache = FrozenPerceptionCache(CACHE)
    lru = PayloadLRU(cache, max_entries=512)
    gt = load_gt(ANNOTATIONS)
    while (chunk := claim(args.output)) is not None:
        directory = args.output / "chunks" / chunk["name"]
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / "records.jsonl"
        temporary = directory / "records.jsonl.partial"
        progress = {"chunk": chunk["name"], "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "completed_records": 0, "expected_records": chunk["decision_count"], "started_utc": now()}
        last_write = 0.0
        started = time.monotonic()
        try:
            state, metadata = load_state_snapshot(Path(chunk["snapshot"]))
            if digest(chunk["snapshot"]) != chunk["snapshot_sha256"].removeprefix("sha256:"):
                raise RuntimeError("snapshot hash mismatch")
            if metadata["source_commit"] != plan["binding"]["source_commit"] or metadata["key_start"] != chunk["key_start"]:
                raise RuntimeError("snapshot source/range mismatch")
            with temporary.open("w") as handle:
                def publish(fields=None, force=False):
                    nonlocal last_write
                    progress.update(fields or {})
                    elapsed = time.monotonic()
                    if force or elapsed - last_write >= 2:
                        handle.flush()
                        progress.update(updated_utc=now(), elapsed_seconds=elapsed - started)
                        atomic(directory / "progress.json", progress)
                        last_write = elapsed
                def sink(record):
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    progress["completed_records"] += 1
                    publish({"phase": "record_complete"})
                publish({"phase": "starting"}, force=True)
                _, stats, skipped = build_v2_records(
                    trace=trace_path(chunk["video_id"]), cache_root=CACHE, annotations=ANNOTATIONS,
                    checkpoint_hash=plan["binding"]["gmt_checkpoint_sha256"], horizon=8,
                    association_backend="formal_gmt_transformer", engine=model, cache_obj=lru, gt_bundle=gt,
                    cache_keys_by_video={chunk["video_id"]: [k for k in cache.keys() if k[0] == chunk["video_id"]]},
                    record_sink=sink, video_ids=[chunk["video_id"]], initial_state=state,
                    key_start_index=chunk["key_start"], key_end_index=chunk["key_end"],
                    selected_key_range=(chunk["key_start"], chunk["key_end"]), progress_callback=publish)
                if skipped or stats != chunk["expected_question_counts"] or sum(stats.values()) != chunk["decision_count"]:
                    raise RuntimeError(f"chunk coverage mismatch: {stats}, skipped={skipped}")
                handle.flush()
                os.fsync(handle.fileno())
                publish({"phase": "COMPLETE"}, force=True)
            os.replace(temporary, record_path)
            manifest = {"status": "COMPLETE", "records": sum(stats.values()), "records_by_question": stats,
                        "records_sha256": digest(record_path), "binding": plan["binding"],
                        "chunk": chunk, "completed_utc": now(), "elapsed_seconds": time.monotonic() - started}
            atomic(directory / "manifest.json", manifest)
            def complete(q):
                c = next(c for c in q["chunks"] if c["name"] == chunk["name"])
                c.update(status="COMPLETE", records=manifest["records"], elapsed_seconds=manifest["elapsed_seconds"],
                         completed_utc=now(), records_sha256=manifest["records_sha256"])
            queue_update(args.output, complete)
            print(json.dumps({"phase": "chunk_complete", "chunk": chunk["name"], "records": sum(stats.values())}), flush=True)
        except Exception:
            error = traceback.format_exc()
            atomic(directory / "error.json", {"status": "FAILED", "error": error, "updated_utc": now()})
            def fail(q):
                c = next(c for c in q["chunks"] if c["name"] == chunk["name"])
                c.update(status="FAILED", error=error)
            queue_update(args.output, fail)
            raise


def spawn(args, mode, gpu, label, extra=()):
    log = args.output / (label + ".log")
    handle = log.open("a")
    command = [PYTHON, "-u", str(Path(__file__).resolve()), mode, "--output", str(args.output),
               "--videos", *map(str, args.videos), "--chunk-records", str(args.chunk_records)]
    if args.frames:
        command += ["--frames", *map(str, args.frames)]
    command += list(extra)
    process = subprocess.Popen(command, cwd=ROOT, env=environment(gpu), stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    return process


def workers(args, gpus):
    def recover(q):
        for c in q["chunks"]:
            if c["status"] not in ("RUNNING", "FAILED"):
                continue
            manifest = args.output / "chunks" / c["name"] / "manifest.json"
            records = manifest.with_name("records.jsonl")
            if manifest.exists() and records.exists():
                m = read(manifest)
                if m["binding"] == q["binding"] and m["records"] == c["decision_count"] and digest(records) == m["records_sha256"]:
                    c.update(status="COMPLETE", records=m["records"], records_sha256=m["records_sha256"])
                    continue
            if c["status"] == "RUNNING":
                try:
                    os.kill(c["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f"another live worker owns {c['name']}")
            c.update(status="PENDING", recovered_utc=now())
    queue_update(args.output, recover)
    processes = [spawn(args, "worker", gpu, f"worker_gpu{gpu}") for gpu in gpus]
    while any(p.poll() is None for p in processes):
        q = read(args.output / "queue.json")
        done = sum(c.get("records", 0) for c in q["chunks"] if c["status"] == "COMPLETE")
        partial = 0
        for c in q["chunks"]:
            p = args.output / "chunks" / c["name"] / "progress.json"
            if c["status"] == "RUNNING" and p.exists():
                partial += read(p).get("completed_records", 0)
        report = {"phase": "building", "completed_records": done + partial,
                  "total_records": sum(c["decision_count"] for c in q["chunks"]),
                  "chunks": dict(Counter(c["status"] for c in q["chunks"])), "updated_utc": now()}
        atomic(args.output / "progress.json", report)
        print(json.dumps(report), flush=True)
        time.sleep(10)
    failures = [p.returncode for p in processes if p.returncode]
    if failures or any(c["status"] != "COMPLETE" for c in read(args.output / "queue.json")["chunks"]):
        raise RuntimeError(f"incomplete/failed chunk workers: {failures}")


def merge(args):
    from build_jev_counterfactual_v2 import UTILITY_DEFINITION
    from jev_intra_video_chunking import semantic_record_key
    plan = read(args.output / "plan.json")
    if plan["binding"] != source_binding():
        raise RuntimeError("merge source/input provenance changed")
    for video in plan["videos"]:
        records = []
        for c in sorted((c for c in plan["chunks"] if c["video_id"] == video), key=lambda c: c["key_start"]):
            directory = args.output / "chunks" / c["name"]
            m = read(directory / "manifest.json")
            if m["status"] != "COMPLETE" or m["binding"] != plan["binding"] or digest(directory / "records.jsonl") != m["records_sha256"]:
                raise RuntimeError("chunk manifest/hash mismatch")
            records.extend(json.loads(l) for l in (directory / "records.jsonl").open())
        records.sort(key=lambda r: r["state"]["online_context"]["event_order"])
        if len(records) != plan["video_metadata"][str(video)]["records"] or len({semantic_record_key(r) for r in records}) != len(records):
            raise RuntimeError("merged record coverage/uniqueness mismatch")
        path = args.output / f"video{video:02d}_records.jsonl"
        temp = path.with_suffix(".jsonl.partial")
        with temp.open("w") as handle:
            for r in records:
                handle.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
        metadata = plan["video_metadata"][str(video)]
        m = {**plan["binding"], "status": "PASS", "created_utc": now(), "trace": metadata["trace"],
             "trace_sha256": metadata["trace_sha256"], "records": len(records),
             "records_by_question": dict(Counter(r["question_type"] for r in records)),
             "records_sha256": digest(path), "skipped_events": 0, "derived_horizons": [8],
             "utility_definition": UTILITY_DEFINITION, "uses_future_gt": True,
             "sampling": False, "truncation": bool(args.frames), "segmented": True,
             "segment_count": sum(c["video_id"] == video for c in plan["chunks"]),
             "resume_unit": "complete chunk from exact OFF/RNG snapshot", "plan_sha256": digest(args.output / "plan.json")}
        atomic(Path(str(path) + ".manifest.json"), m)
        print(json.dumps({"phase": "merged", "video_id": video, "records": len(records)}), flush=True)


def baseline(args):
    from build_jev_counterfactual_v2 import build_v2_records
    plan = read(args.output / "plan.json")
    m = plan["video_metadata"]["1"]
    records, _, skipped = build_v2_records(trace=trace_path(1), cache_root=CACHE, annotations=ANNOTATIONS,
             checkpoint_hash=plan["binding"]["gmt_checkpoint_sha256"], horizon=8,
             association_backend="formal_gmt_transformer", engine=engine(), video_ids=[1],
             key_end_index=m["key_end"], selected_key_range=(m["key_start"], m["key_end"]))
    if skipped:
        raise RuntimeError("baseline skipped events")
    with (args.output / "single_records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")


def verify(args):
    from jev_intra_video_chunking import compare_records_exact
    prepare(args)
    reference = spawn(args, "baseline", args.gpus[-1], "single_reference")
    workers(args, args.gpus[1:3])
    merge(args)
    if reference.wait():
        raise RuntimeError("single reference failed")
    left = [json.loads(l) for l in (args.output / "single_records.jsonl").open()]
    right = [json.loads(l) for l in (args.output / "video01_records.jsonl").open()]
    report = compare_records_exact(left, right)
    report.update(scope="frames 210-216; full trace and cross-boundary H8 futures",
                  source_commit=commit(), question_counts=dict(Counter(r["question_type"] for r in right)))
    report["final_gate"] = {name: report["status"] == "PASS" for name in (
        "raw_canonical_records_exact", "best_actions_exact", "target_probs_within_1e-6",
        "utilities_within_1e-6", "state_features_within_1e-6", "ranges_no_gap_overlap_duplicate",
        "rng_provenance_exact", "canonical_sha_identical")}
    atomic(args.output / "equivalence.json", report)
    if report["status"] != "PASS" or not report["question_counts"].get("REACTIVATION_DECISION"):
        raise RuntimeError("formal segmentation equivalence/coverage failed")


def command(args, name, command, gpu):
    log = args.output / (name + ".log")
    with log.open("a") as f:
        result = subprocess.run([str(x) for x in command], cwd=ROOT, env=environment(gpu), stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed; exit={result.returncode}; log={log}")


def aftercare(args):
    reports = args.output / "reports"
    reports.mkdir(exist_ok=True)
    for video in args.videos:
        records = args.output / f"video{video:02d}_records.jsonl"
        m = read(Path(str(records) + ".manifest.json"))
        command(args, f"provenance{video}", [PYTHON, ROOT / "reproduction_tools/validate_jev_video_artifact.py",
                "--manifest", str(records) + ".manifest.json", "--records", records,
                "--output", reports / f"provenance{video}.json", "--expected-records", m["records"], "--horizon", 8], args.gpus[0])
        command(args, f"parity{video}", [PYTHON, ROOT / "reproduction_tools/run_jev_runtime_feature_parity.py",
                "--video-id", video, "--trace", trace_path(video), "--records", records,
                "--output", reports / f"parity{video}.json", "--device", "cuda:0", "--tolerance", "2e-5"], args.gpus[0])
    records = args.output / "video01_records.jsonl"
    command(args, "candidate_parity", [PYTHON, ROOT / "reproduction_tools/compare_reactivation_candidates.py",
            "--video-id", 1, "--native-trace", BASE / "native_video1_corrected_v4b/native_off_trace_video01.jsonl",
            "--records", records, "--output", reports / "candidate_parity.json",
            "--replay-root", args.output / "candidate_replay", "--device", "cuda:0",
            "--max-frame", 1000000, "--tolerance", "2e-5"], args.gpus[0])
    command(args, "stability", [PYTHON, ROOT / "reproduction_tools/run_jev_feature_parity_stability.py",
            "--video-id", 1, "--trace", trace_path(1), "--records", records,
            "--output", reports / "stability.json", "--runtime-root", args.output / "stability",
            "--device", "cuda:0", "--max-frame", 1000000, "--repetitions", 3, "--tolerance", "2e-5"], args.gpus[0])
    training = args.output / "small_h8_training_current_head_video06_video07"
    dataset = training / "compact_v1"
    split = training / "policy_split.json"
    methods = training / "methods_v1"
    training.mkdir(exist_ok=True)
    command(args, "compact", [PYTHON, ROOT / "reproduction_tools/jev_compact_dataset.py", "--input",
            args.output / "video06_records.jsonl", args.output / "video07_records.jsonl", "--output", dataset], args.gpus[0])
    command(args, "split", [PYTHON, ROOT / "reproduction_tools/create_jev_policy_split.py", "--input",
            args.output / "video06_records.jsonl", args.output / "video07_records.jsonl", "--output", split,
            "--seed", 20261003, "--val-fraction", 0.2], args.gpus[0])
    jobs = []
    for (method, width), gpu in zip((("question_threshold", 140), ("question_conditioned_mlp", 139), ("jev", 128)), args.gpus):
        f = (args.output / (method + ".log")).open("a")
        p = subprocess.Popen([PYTHON, "-u", str(ROOT / "reproduction_tools/run_jev_three_way_method.py"),
                "--dataset", str(dataset), "--split-manifest", str(split), "--output", str(methods / method),
                "--model", method, "--hidden-dim", str(width), "--epochs", "20", "--batch-size", "128",
                "--lr", "0.001", "--device", "cuda:0"], cwd=ROOT, env=environment(gpu), stdout=f, stderr=subprocess.STDOUT)
        f.close(); jobs.append(p)
    codes = [p.wait() for p in jobs]
    if any(codes):
        raise RuntimeError("three-way training failed")
    command(args, "aggregate", [PYTHON, ROOT / "reproduction_tools/aggregate_jev_three_way_compact.py",
            "--dataset", dataset, "--split-manifest", split, "--methods-root", methods,
            "--output", reports / "training.json", "--markdown", args.output / "three_way.md"], args.gpus[0])
    command(args, "closed_loop", [PYTHON, ROOT / "reproduction_tools/run_corrected_v4_tracking.py",
            "--video-id", 1, "--trace", trace_path(1), "--records", records,
            "--methods-root", methods, "--output-root", args.output / "closed_loop",
            "--output-report", reports / "tracking.json", "--provenance-report", reports / "provenance1.json",
            "--candidate-report", reports / "candidate_parity.json", "--feature-parity-report", reports / "parity1.json",
            "--stability-report", reports / "stability.json", "--formal-report", args.output / "probe/equivalence.json",
            "--training-report", reports / "training.json", "--device", "cuda:0", "--tolerance", "2e-5"], args.gpus[0])


def run(args):
    probe = argparse.Namespace(**vars(args))
    probe.output = args.output / "probe"
    probe.videos, probe.frames, probe.chunk_records = [1], [210, 216], 25
    if not (probe.output / "equivalence.json").exists():
        verify(probe)
    if read(probe.output / "equivalence.json")["status"] != "PASS":
        raise RuntimeError("segmentation equivalence not passed")
    prepare(args)
    workers(args, args.gpus)
    merge(args)
    atomic(args.output / "progress.json", {"phase": "acceptance", "updated_utc": now()})
    aftercare(args)
    atomic(args.output / "progress.json", {"phase": "COMPLETE", "updated_utc": now(), "report": str(args.output / "reports/tracking.json")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("run", "prepare", "worker", "merge", "baseline", "verify", "aftercare"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--videos", type=int, nargs="+", default=[1, 6, 7])
    p.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 5, 6, 7, 8, 9])
    p.add_argument("--chunk-records", type=int, default=200)
    p.add_argument("--frames", type=int, nargs=2)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in ("run", "prepare", "verify"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    guard = None
    if args.mode in ("run", "verify"):
        guard = (args.output / "driver.lock").open("a+")
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        globals()[args.mode](args)
    except Exception:
        atomic(args.output / f"failure_{args.mode}_{os.getpid()}.json", {"status": "FAILED", "updated_utc": now(), "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
