#!/usr/bin/env python3
"""Convert the repository's released XYXY View*.txt output to audit JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-root',type=Path,required=True); ap.add_argument('--annotations',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    data=json.loads(args.annotations.read_text()); by_key={}
    for im in data['images']:
        v=next(x for x in data['videos'] if x['id']==im['video_id'])
        by_key[(v['file_name'],int(im['view_id']),int(im['frame_id']))]=im['id']
    out=[]
    for scene_dir in sorted(args.raw_root.iterdir()):
        if not scene_dir.is_dir(): continue
        for fp in sorted(scene_dir.glob('View*.txt')):
            view=int(fp.stem.replace('View',''))
            for line in fp.read_text().splitlines():
                if not line.strip(): continue
                z=[float(x) for x in line.split(',')]
                if len(z)<6: continue
                frame,tid,x1,y1,x2,y2=z[:6]; image_id=by_key.get((scene_dir.name,view,int(frame)))
                if image_id is None: continue
                score=float(z[6]) if len(z)>6 else 1.0
                out.append({'image_id':int(image_id),'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'score':score,'track_id':int(tid)})
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,separators=(',',':'))+'\n'); print(json.dumps({'predictions':len(out),'output':str(args.output)},indent=2))
if __name__=='__main__': main()
