import json
from common import *

def main():
    base=RUNTIME/'baseline/B0_smoke60/video_01.json';reports={}
    for tag in ['OFF_after_hook','ZERO_after_hook']:
        path=RUNTIME/'baseline'/tag/'video_01.json'
        a,b=json.loads(base.read_text()),json.loads(path.read_text());ok=a==b and sha256(base)==sha256(path)
        reports[tag]=dict(exact_predictions=ok,rows=len(b),reference_sha256=sha256(base),candidate_sha256=sha256(path),comparison='all serialized detection boxes, scores, classes, image IDs and track IDs; 60 synchronized frame groups')
        if not ok:raise AssertionError(tag)
    save_json(ROOT/'results/off_zero_parity.json',dict(status='PASS',reports=reports,full_video_parity_pending=True))
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
