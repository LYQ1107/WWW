#!/usr/bin/env python3
"""Exercise unchanged native evaluators on generated, non-dataset tracks only.

This is an engineering smoke test, not a VisionTrack TRAIN/TEST experiment.
No original images, annotations, predictions, model weights or authorization
subsets are consumed. It also verifies why per-camera IDSW cannot replace
global cross-view identity agreement.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from worldmot.evaluation.official_fullvideo import write_mot, native_tracking_sources, sha
from worldmot.runtime.world_object_v4_test_contract import install_no_gt_guard
from worldmot.runtime.world_object_v5_contract import v5_source_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    guard = install_no_gt_guard()
    source = v5_source_fingerprint()
    kit = a.repo / "MOTChallengeEvalKit_cv_test/matlab_devkit"
    before = {str(path.relative_to(kit)): sha(path) for path in native_tracking_sources(kit)}
    jobs, counts = [], {}
    cases = ("perfect_shared_ID", "stable_view_specific_IDs", "shared_temporal_ID_change")
    sys.path.insert(0, str(a.repo / "TrackEval"))
    import trackeval
    for case in cases:
        gt, pred = {}, {}
        folder = a.output / case
        for camera in (1, 2):
            name = f"synthetic_View{camera}"
            gt[name], pred[name] = [], []
            for tick in range(1, 5):
                identity = 10
                if case == "stable_view_specific_IDs":
                    identity += camera
                if case == "shared_temporal_ID_change" and tick >= 3:
                    identity = 20
                bbox = [100. + 50 * camera, 80., 40., 90.]
                gt[name].append([tick, 1, *bbox, 1, 1, 1, -1])
                pred[name].append([tick, identity, *bbox, 1, -1, -1, -1])
            write_mot(folder / "gt" / name / "gt/gt.txt", gt[name])
            write_mot(folder / "trackers/Synthetic/data" / f"{name}.txt", pred[name])
        ec = trackeval.Evaluator.get_default_eval_config()
        ec.update(USE_PARALLEL=False, PRINT_CONFIG=False, PRINT_RESULTS=False, PLOT_CURVES=False,
                  OUTPUT_FOLDER=str(folder / "trackeval_output"))
        dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dc.update(GT_FOLDER=str(folder / "gt"), TRACKERS_FOLDER=str(folder / "trackers"),
                  TRACKERS_TO_EVAL=["Synthetic"], BENCHMARK="SYNTHETIC_ONLY", SPLIT_TO_EVAL="synthetic",
                  SKIP_SPLIT_FOL=True, SEQ_INFO={name: 4 for name in gt}, PRINT_CONFIG=False,
                  DO_PREPROC=False, OUTPUT_FOLDER=str(folder / "trackeval_output"))
        dataset = trackeval.datasets.MotChallenge2DBox(dc)
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
        values, _ = trackeval.Evaluator(ec).evaluate([dataset], metrics)
        row = values[dataset.get_name()]["Synthetic"]["COMBINED_SEQ"]["pedestrian"]
        counts[case] = {"TrackEval_IDSW": int(row["CLEAR"]["IDSW"]),
                        "TrackEval_IDF1": 100 * float(row["Identity"]["IDF1"]),
                        "TrackEval_TP": int(row["CLEAR"]["CLR_TP"])}
        for convention in ("sequential", "interleaved"):
            parts = {"gt": [], "prediction": []}
            offset = 0
            for camera, name in enumerate(sorted(gt)):
                for rows, key in ((gt[name], "gt"), (pred[name], "prediction")):
                    for item in rows:
                        frame = int(item[0]) + offset if convention == "sequential" else 2 * int(item[0]) + camera
                        parts[key].append([frame, *item[1:6], -1, -1, -1, -1])
                offset += 5
            paths = {key: folder / convention / f"{key}.txt" for key in parts}
            for key, rows in parts.items():
                write_mot(paths[key], rows)
            jobs.append({"scene": case, "convention": convention, "groundtruth": str(paths["gt"].resolve()),
                         "prediction": str(paths["prediction"].resolve()), "gt_directory": str(paths["gt"].parent.resolve())})
    native = a.output / "native_synthetic.json"
    config = a.output / "CONFIG.json"
    config.write_text(json.dumps({"kit": str(kit.resolve()), "jobs": jobs, "output": str(native.resolve()), "benchmark": "VisionTrack"}))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "WORLDMOT_MATLAB_CONFIG": str(config.resolve()),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("MKL_THREADING_LAYER", None)
    driver = a.repo / "worldmot/scripts/official_worldmot_metrics.m"
    with (a.output / "matlab.log").open("x") as log:
        process = subprocess.Popen(["/data1/liuyeqiang/MATLAB/R2020a/bin/matlab", "-batch", "run('" + str(driver.resolve()) + "')"],
                                   cwd=a.repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (a.output / "matlab_process.json").write_text(json.dumps({"pid": process.pid, "synthetic_only": True}))
        code = process.wait()
    if code:
        raise RuntimeError("SYNTHETIC_NATIVE_ENGINE_FAILED_" + str(code))
    data = json.loads(native.read_text())
    for item in data["results"]:
        if item["convention"] == "sequential":
            identity = item["Identity"]
            counts[item["scene"]]["native_CVIDF1_synthetic"] = 200 * identity["IDTP"] / (identity["n_gt"] + identity["n_tr"])
        else:
            counts[item["scene"]]["native_CLEAR_IDSW_interleaved"] = item["CLEAR"]["id_switches"]
    expected = {"perfect_shared_ID": (0, 100., 100.), "stable_view_specific_IDs": (0, 100., 50.),
                "shared_temporal_ID_change": (2, 50., 50.)}
    for case, (switches, idf1, cvidf1) in expected.items():
        actual = counts[case]
        if (actual["TrackEval_IDSW"] != switches or abs(actual["TrackEval_IDF1"] - idf1) > 1e-8
                or abs(actual["native_CVIDF1_synthetic"] - cvidf1) > 1e-8 or actual["TrackEval_TP"] != 8):
            raise ValueError("SYNTHETIC_PROTOCOL_EXPECTATION_FAILED: " + case + " " + str(actual))
    after = {str(path.relative_to(kit)): sha(path) for path in native_tracking_sources(kit)}
    if before != after or guard["blocked_GT_reads"] or v5_source_fingerprint() != source:
        raise ValueError("TOOL_OR_MODEL_SOURCE_CHANGED_OR_ORIGINAL_GT_ACCESSED")
    report = {"status": "PASSED_SYNTHETIC_TRACKEVAL_AND_NATIVE_MATLAB_PROTOCOL", "cases": counts,
              "original_TRAIN_or_TEST_data_used": False, "formal_V5_score": False, "model_used": False,
              "native_MATLAB_version": data["version"], "native_function": data["native_official_function"],
              "toolkit_sha256": before, "driver_sha256": sha(driver), "analysis_sha256": sha(__file__),
              "v5_source_sha256": source, "UTC": datetime.now(timezone.utc).isoformat(),
              "conclusion": "Stable camera-specific IDs can have per-camera IDSW=0 while native global identity agreement is 50%; these are synthetic values only."}
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": counts}), flush=True)


if __name__ == "__main__":
    main()
