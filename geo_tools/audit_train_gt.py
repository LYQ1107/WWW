"""Audit complete original TRAIN before result selection; no label repair."""
import collections
from common import *

def main():
    data=train_data();rows=[]
    for v in data['videos']:
        for view in range(1,v['view_num']+1):
            path=DATA/'train'/f"{v['file_name']}_View{view}"/'gt/gt.txt'
            lines=[r.strip().split(',') for r in path.read_text().splitlines() if r.strip()];groups=collections.defaultdict(list)
            for r in lines:groups[(int(float(r[0])),int(float(r[1])))].append(r)
            dup=[r for r in groups.values() if len(r)>1]
            rows.append(dict(video_id=v['id'],scene=v['file_name'],view_id=view,raw_rows=len(lines),
                duplicate_identity_rows=sum(len(r)-1 for r in dup),conflicting_bbox_groups=sum(len({tuple(x[2:6]) for x in r})>1 for r in dup),
                exact_duplicate_groups=sum(len({tuple(x) for x in r})==1 for r in dup),raw_sha256=sha256(path)))
    save_json(ROOT/'results/train_gt_quality_audit.json',dict(source_split='original TRAIN',label_policy='raw rows preserved, no repair',rows=rows))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
