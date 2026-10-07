from pathlib import Path
import argparse,fcntl,inspect,json,os,sys,time,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import mps_launcher
entry=mps_launcher.entry
sys.path.insert(0,str(entry.ROOT/"reproduction_tools"))
from run_corrected_v4_tracking import gate_snapshot
args=argparse.Namespace(output=HERE,gpus=[2,3,5],videos=[1,6,7],frames=None,chunk_records=200)
lock=(HERE/"experiments.lock").open("a+")
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    changed=entry.subprocess.check_output(["git","diff","--name-only"],cwd=entry.ROOT,text=True).splitlines()
    generated="reports/JEV_RUNTIME_STATE_V3/STATE_FEATURE_PARITY.json"
    if any(path!=generated for path in changed):raise RuntimeError("non-output tracked files changed")
    original_run=entry.subprocess.run
    def source_guard(cmd,*a,**kw):
        if cmd==["git","diff","--exit-code"]:
            cmd=cmd+["--",".",":(exclude)"+generated]
        return original_run(cmd,*a,**kw)
    entry.subprocess.run=source_guard
    try:binding=entry.source_binding()
    finally:entry.subprocess.run=original_run
    if changed:
        entry.atomic(HERE/"generated_report_source_guard.json",{"source_commit":binding["source_commit"],"ignored_generated_output":generated,"generated_output_sha256":entry.digest(entry.ROOT/generated),"all_other_tracked_files_unchanged":True})
    plan=entry.read(HERE/"plan.json")
    if binding!=plan["binding"]:raise RuntimeError("source/input binding changed")
    paths={"provenance":HERE/"reports/provenance1.json","candidate":HERE/"reports/candidate_parity.json","feature_parity":HERE/"reports/parity1.json","stability":HERE/"reports/stability.json","formal":HERE/"probe/equivalence.json","training":HERE/"reports/training.json"}
    while not paths["stability"].exists():
        print(json.dumps({"phase":"waiting_for_existing_stability"}),flush=True)
        time.sleep(10)
    gates,failures=gate_snapshot(paths,2e-5)
    failures=[f for f in failures if f!="missing_or_invalid_training"]
    if failures:raise RuntimeError(str(failures))
    for video in (1,6,7):
        for kind in ("provenance","parity"):
            if entry.read(HERE/"reports"/f"{kind}{video}.json")["status"]!="PASS":raise RuntimeError("video gate failed")
        m=entry.read(HERE/f"video{video:02d}_records.jsonl.manifest.json")
        if m["source_commit"]!=binding["source_commit"] or m["records_sha256"]!=entry.digest(HERE/f"video{video:02d}_records.jsonl"):raise RuntimeError("record binding failed")
    for name in ("late_probe/equivalence.json","mps_equivalence.json","mps_late_equivalence.json"):
        if entry.read(HERE/name)["status"]!="PASS":raise RuntimeError("equivalence gate failed")
    stability=entry.read(paths["stability"]);candidate=entry.read(paths["candidate"])
    for report,field,path in ((stability,"records_sha256",HERE/"video01_records.jsonl"),(stability,"trace_sha256",entry.trace_path(1)),(candidate,"replay_records_sha256",HERE/"video01_records.jsonl"),(candidate,"native_trace_sha256",entry.trace_path(1))):
        if report[field].removeprefix("sha256:")!=entry.digest(path):raise RuntimeError("gate input hash mismatch")
    body=inspect.getsource(entry.aftercare)
    if paths["training"].exists() and entry.read(paths["training"])["status"]=="PASS":
        tail=body[body.index('    command(args, "closed_loop"'):]
    else:
        tail=body[body.index('    training = args.output / "small_h8_training_current_head_video06_video07"'):]
    tail=tail.replace("    jobs = []",'    atomic(args.output / "progress.json", {"phase":"training","updated_utc":now(),"models":["Threshold","MLP","JEV"]})\n    print(json.dumps({"phase":"training","models":["Threshold","MLP","JEV"]}),flush=True)\n    jobs = []')
    original_command=entry.command
    def command(args,name,cmd,gpu):
        entry.atomic(HERE/"progress.json",{"phase":name,"updated_utc":entry.now(),"driver_pid":os.getpid()})
        print(json.dumps({"phase":name}),flush=True)
        original_command(args,name,cmd,gpu)
    entry.command=command
    exec('def finish_experiments(args):\n    reports=args.output / "reports"\n    records=args.output / "video01_records.jsonl"\n    methods=args.output / "small_h8_training_current_head_video06_video07/methods_v1"\n'+tail,vars(entry))
    entry.finish_experiments(args)
    tracking=entry.read(HERE/"reports/tracking.json")
    if tracking["status"]!="PASS":raise RuntimeError("tracking execution failed")
    entry.atomic(HERE/"progress.json",{"phase":"COMPLETE","updated_utc":entry.now(),"report":str(HERE/"reports/tracking.json")})
    print(json.dumps({"status":"COMPLETE","report":str(HERE/"reports/tracking.json")}),flush=True)
except BaseException:
    entry.atomic(HERE/"failure_continuation.json",{"status":"FAILED","error":traceback.format_exc(),"updated_utc":entry.now()})
    raise
