"""Engineering dry run of complete query schedules; no eligible/TRAIN gradient claim."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
from pirga_indexed_store import epoch_order,coverage
from audit_pirga_fulltrain import FIT,checked


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();pins=json.loads(a.source_manifest.read_text())['artifacts'];by=defaultdict(list);people={};states={};person_counts=Counter();started=time.monotonic()
    for scene in FIT:
        path=a.runtime/'full_train_audit_v1'/f'scene{scene}_full_v1/events.jsonl.gz';checked(path,pins[str(path)])
        with gzip.open(path,'rt') as f:
            for line in f:
                r=json.loads(line);k=tuple(r['key']);by[scene].append(k);people[k]=r['query_gt_identity'];states[k]=r['actual_primary_state']
                if people[k] is not None:person_counts[(scene,people[k])]+=1
    allkeys=sum(by.values(),[]);reports=[]
    for epoch in range(1,6):
        order=list(epoch_order(by,20261016,epoch));report=coverage(allkeys,order)
        if not report['full_epoch_pass']:raise ValueError('Schedule coverage failed')
        report.update(epoch=epoch,order_sha256=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest(),query_metadata_only=True,actual_gradient_updates=0,ranking_eligibility_evaluated=False,
                      scene_person_groups=len(person_counts),states=dict(Counter(states[k] for k in order)))
        reports.append(report)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir();(a.out/'TRAINING_EPOCH_COVERAGE.json').write_text(json.dumps(dict(status='PASS_ENGINEERING_ALL_QUERY_SCHEDULE_ONLY',seed=20261016,
        epochs=reports,supervision_eligible_count=None,actual_trained_queries=0,actual_gradient_updates=0,
        note='All154911 real physical query keys exercise scheduler. Not an epoch of training: complete native inputs and eligibility still missing.',wall_seconds=time.monotonic()-started),indent=2)+'\n')
    import csv
    with open(a.out/'PER_PERSON_AND_SCENE_INPUT_COUNTS.csv','w') as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(['scene','GT_person_offline_only','actual_query_exposures','trained_query_exposures'])
        for (scene,person),count in sorted(person_counts.items()):w.writerow([scene,person,count,0])

if __name__=='__main__':main()
