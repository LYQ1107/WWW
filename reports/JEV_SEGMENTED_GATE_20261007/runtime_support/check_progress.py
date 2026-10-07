from pathlib import Path
from collections import Counter
import argparse, datetime as dt, json, os, time
ROOT=Path(__file__).resolve().parent

def load(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return None

def show():
    print(dt.datetime.now().astimezone().isoformat(), flush=True)
    for label, directory in (("Probe", ROOT/"probe"), ("Late probe", ROOT/"late_probe"), ("Full build", ROOT)):
        progress=load(directory/"progress.json")
        if progress:
            queue=load(directory/"queue.json")
            if queue:
                partial=sum((load(directory/"chunks"/c["name"]/"progress.json") or {}).get("completed_records",0) for c in queue["chunks"] if c["status"]=="RUNNING")
                progress["completed_records"]=sum(c.get("records",0) for c in queue["chunks"] if c["status"]=="COMPLETE")+partial
                progress["chunks"]=dict(Counter(c["status"] for c in queue["chunks"]))
            print(label, json.dumps(progress, ensure_ascii=False), flush=True)
            if queue:
                for c in queue["chunks"]:
                    if c["status"]!="RUNNING": continue
                    cp=load(directory/"chunks"/c["name"]/"progress.json")
                    age=round(time.time()-(directory/"chunks"/c["name"]/"progress.json").stat().st_mtime,1) if cp else None
                    try: os.kill(c["pid"],0); live=True
                    except ProcessLookupError: live=False
                    print(json.dumps({"chunk":c["name"],"gpu":c.get("gpu"),"pid":c["pid"],"process_alive":live,"last_progress_age_seconds":age,"progress":cp},ensure_ascii=False),flush=True)
    for relative in ("probe/equivalence.json", "probe/native_candidate_probe.json", "late_probe/equivalence.json", "late_probe/native_candidate_probe.json", "reports/training.json", "reports/tracking.json"):
        r=load(ROOT/relative)
        if r: print(relative, r.get("status"), flush=True)
    for f in sorted(ROOT.glob("failure_*.json")): print("FAILED",f,flush=True)
    for f in sorted((ROOT/"reports").glob("*.json")):
        r=load(f)
        if r: print("Report",f.name,r.get("status"),flush=True)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try: args=(proc/"cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError): continue
        if not any(a.startswith(str(ROOT)) for a in args): continue
        task=next((Path(a).name for a in args if a.endswith(".py") and "reproduction_tools" in a), None)
        if task and task!="run_segmented_small_gate.py": print("Active",proc.name,task,flush=True)

p=argparse.ArgumentParser()
p.add_argument("--watch",type=float,default=0)
a=p.parse_args()
while True:
    show()
    if not a.watch: break
    time.sleep(a.watch)
