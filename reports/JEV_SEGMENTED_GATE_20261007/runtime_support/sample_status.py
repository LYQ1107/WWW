from pathlib import Path
from collections import Counter
import datetime as dt,json,time
ROOT=Path(__file__).resolve().parent
q=json.loads((ROOT/"queue.json").read_text())
done=sum(c.get("records",0) for c in q["chunks"] if c["status"]=="COMPLETE")
partial=0;ages=[]
for c in q["chunks"]:
    if c["status"]!="RUNNING":continue
    p=ROOT/"chunks"/c["name"]/"progress.json"
    if p.exists():
        partial+=json.loads(p.read_text()).get("completed_records",0)
        ages.append(time.time()-p.stat().st_mtime)
summary={"utc":dt.datetime.now(dt.timezone.utc).isoformat(),"epoch":time.time(),"completed_records":done+partial,"finalized_records":done,"total_records":sum(c["decision_count"] for c in q["chunks"]),"segments":dict(Counter(c["status"] for c in q["chunks"])),"max_progress_age_seconds":round(max(ages,default=0),1)}
progress=json.loads((ROOT/"progress.json").read_text())
summary["phase"]=progress["phase"]
path=ROOT/"performance_samples.jsonl"
previous=json.loads(path.read_text().splitlines()[-1]) if path.exists() else None
if previous and summary["completed_records"]>=previous["completed_records"]:
    summary["recent_records_per_second"]=round((summary["completed_records"]-previous["completed_records"])/(summary["epoch"]-previous["epoch"]),3)
with path.open("a") as f:f.write(json.dumps(summary)+"\n")
print(json.dumps(summary,ensure_ascii=False))
for c in q["chunks"]:
    if c["status"]=="FAILED":print("FAILED",c["name"],c.get("error"))
for p in ROOT.glob("failure_*.json"):print("FAILURE",p.name,p.read_text()[-1200:])
if summary["phase"]!="building":
    for p in sorted((ROOT/"reports").glob("*.json")):
        r=json.loads(p.read_text());print(p.name,r.get("status"))
