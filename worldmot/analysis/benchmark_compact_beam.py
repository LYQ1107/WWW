#!/usr/bin/env python3
"""Representation-only CPU microbenchmark; never model or end-to-end FPS."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from worldmot.tracking.joint_world_object_assignment import AssignmentOption,JointWorldObjectAssignment
from worldmot.tracking.world_object_config import WorldObjectConfig


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    reference_commit="e6cc4d7c017fd15e0a131e495c4f452f96526efb"
    path="worldmot/worldmot/tracking/joint_world_object_assignment.py"
    source=subprocess.check_output(["git","show",reference_commit+":"+path]).decode()
    tree=ast.parse(source)
    method=next(node for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=="JointWorldObjectAssignment"
                for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=="_solve_component")
    namespace={"math":math};exec(compile(ast.Module(body=[method],type_ignores=[]),"frozen_beam","exec"),namespace)
    frozen=namespace["_solve_component"];solver=JointWorldObjectAssignment(WorldObjectConfig(1024))
    rng=np.random.default_rng(20261025);rows=[]
    for m,n in ((8,20),(30,60),(60,100),(90,120)):
        groups=[]
        for q in range(m):
            left=rng.choice(n//2,4,replace=False).tolist();right=rng.choice(np.arange(n//2,n),4,replace=False).tolist()
            options=[AssignmentOption(q+1,(index,),float(rng.uniform(.5,1.2)),"SINGLE") for index in left+right]
            options.extend(AssignmentOption(q+1,(x,y),float(rng.uniform(1.5,2.5)),"PAIR") for x in left for y in right)
            groups.append(options)
        old=[];new=[]
        for _ in range(3):
            t=time.perf_counter();expected,ea=frozen(solver,groups);old.append(time.perf_counter()-t)
            t=time.perf_counter();actual,aa=solver._solve_component(groups);new.append(time.perf_counter()-t)
            assert actual==expected and aa==ea
        rows.append({"history":m,"observations":n,"reference_p50_s":statistics.median(old),
            "compact_p50_s":statistics.median(new),"representation_speedup":statistics.median(old)/statistics.median(new),
            "selected_options_and_complete_audit_identical":True,"expanded_states":aa["expanded_states"]})
    report={"status":"CPU_REPRESENTATION_EQUIVALENCE_PASS","model_or_E2E_FPS":False,
        "reference_commit":reference_commit,"reference_module_sha256":hashlib.sha256(source.encode()).hexdigest(),
        "current_module_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),"rows":rows,
        "scope":"Same supplied hypothesis groups only. R2 deliberately changes proposal policy outside this solver routine."}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");print(json.dumps(report))


if __name__=="__main__":main()
