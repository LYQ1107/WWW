"""Actual native metric-tool parse smoke, synthetic result never scientific data."""
import argparse
import json
from pathlib import Path
from cir_evaluation import evaluate_trackeval,evaluate_matlab
from common import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    gt={f'smoke_View{camera}':[[frame,1,10,20,30,40,1,1,1] for frame in (1,2)] for camera in (1,2)}
    pred={name:[r[:6]+[.9,-1,-1,-1] for r in rows] for name,rows in gt.items()}
    track=evaluate_trackeval(gt,pred,{name:2 for name in gt},a.out/'trackeval')
    official=evaluate_matlab({'smoke':list(gt)},gt,pred,a.out/'matlab')
    assert track['combined']['HOTA']==100 and track['combined']['IDF1']==100
    assert official['combined']['CVIDF1']==100 and official['combined']['CVMA']==100
    result=dict(status='PASS_SYNTHETIC_OFFICIAL_PARSER_SMOKE',scientific_result=False,
                TrackEval=track,official_MATLAB=official)
    save_json(a.out/'RESULT.json',result);print(json.dumps(dict(status=result['status'],TrackEval=track['combined'],official=official['combined'])),flush=True)


if __name__=='__main__':main()
