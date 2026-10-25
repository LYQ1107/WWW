#!/usr/bin/env python3
"""Run only registered CPU synthetic tests and AST parsing; produce factual JSON."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import time

TESTS=["test_worldobject_feature_contract.py","test_worldobject_camera_depth.py",
       "test_worldobject_joint_assignment.py","test_worldobject_memory.py",
       "test_worldobject_identity_loss.py","test_worldobject_ablation.py",
       "test_worldobject_causal_runtime.py","test_worldobject_delivery_contract.py",
       "test_worldobject_core.py","test_gmt_detection_cache.py",
       "test_world_alignment.py","test_projection.py","test_gmt_box_storage.py"]


def main():
    p=argparse.ArgumentParser();p.add_argument("--report",required=True,type=Path);a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                      MKL_THREADING_LAYER="GNU")
    import pytest
    root=Path(__file__).resolve().parents[1];repo=root.parent
    files=sorted(list((root/"worldmot").rglob("*.py"))+list((root/"scripts").glob("*.py")))
    syntax=[]
    for path in files:
        content=path.read_bytes()
        try:ast.parse(content,filename=str(path))
        except SyntaxError as exc:syntax.append({"file":str(path.relative_to(repo)),"error":str(exc)})
    class Results:
        def __init__(self):self.cases=[]
        def pytest_runtest_logreport(self,report):
            if report.when=="call" or (report.when in ("setup","teardown") and report.failed):
                self.cases.append({"test":report.nodeid,"outcome":report.outcome,
                                   "failure":str(report.longrepr) if report.failed else None})
    plugin=Results();start=time.perf_counter()
    rc=pytest.main([str(root/"tests"/test) for test in TESTS]+["-q","--tb=short","-p","no:cacheprovider"],plugins=[plugin])
    counts=Counter(case["outcome"] for case in plugin.cases)
    from worldmot.runtime.experiment_authorization import source_fingerprint,source_commit
    report={"schema":"worldobject_cpu_synthetic_test_report_v3","source_commit":source_commit(),"source_sha256":source_fingerprint(),
        "syntax":{"parsed_files":len(files),"errors":syntax},"tests":plugin.cases,"counts":dict(counts),
        "pytest_exit_code":int(rc),"duration_s":time.perf_counter()-start,
        "status":"PASS" if rc==0 and not syntax else "FAIL", "validation_scope":"CPU_SYNTHETIC_ONLY",
        "CUDA_VISIBLE_DEVICES":"","real_dataset_reads":False,"pretrained_weight_loads":False,
        "MKL_THREADING_LAYER":"GNU","KMP_DUPLICATE_LIB_OK_used":False,
        "real_training_inference_evaluation_started":False,"accuracy_metrics_generated":False}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    raise SystemExit(0 if report["status"]=="PASS" else 1)


if __name__=="__main__":
    main()
