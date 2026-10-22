#!/usr/bin/env python3
"""Export all complete camera streams, including scenes with invalid local GT."""
import argparse
import json
from pathlib import Path
from worldmot.evaluation.official_fullvideo import sha, validate, write_mot


def export(metadata, predictions, output, box_source='tracks_2d_by_camera'):
    data=json.loads(Path(metadata).read_text())
    if data.get('annotations'):raise ValueError('export accepts metadata only')
    root=Path(predictions);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for video in data['videos']:
        sid=video['id'];directory=root/f'scene_{sid:02d}'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest.get('status')!='COMPLETE' or not manifest.get('full_video'):raise RuntimeError('incomplete video '+str(sid))
        records=[r for r in data['images'] if r['video_id']==sid]
        expected=sorted({int(r['frame_id']) for r in records})
        path=directory/'tracks.jsonl';rows=[json.loads(line) for line in path.open() if line.strip()]
        if [int(r['timestamp']) for r in rows]!=expected:raise RuntimeError('incomplete timestamp coverage '+str(sid))
        counts={}
        for view in sorted({int(r['view_id']) for r in records}):
            boxes=[]
            for row in rows:
                if box_source not in row:raise RuntimeError('missing box source '+box_source)
                for obj in row[box_source].get(str(view),[]):
                    if not obj.get('visible',True) or not obj.get('evaluation_eligible',True):continue
                    bbox=obj.get('bbox_xywh')
                    if bbox is None:continue
                    boxes.append([row['timestamp'],obj['global_id'],*bbox,obj.get('score',1.),-1,-1,-1])
            name=f'{video["file_name"]}_View{view}'
            validate(boxes,name);write_mot(out/f'{name}.txt',boxes);counts[str(view)]=len(boxes)
        summaries.append({'scene':sid,'frames':len(rows),'source_sha256':sha(path),'boxes_by_camera':counts})
    report={'status':'COMPLETE_ALL_IMAGE_SEQUENCES','metadata_sha256':sha(metadata),'box_source':box_source,
        'GT_read':False,'scenes':summaries,'timestamps':sum(r['frames'] for r in summaries),
        'exporter_sha256':sha(__file__)}
    (out/'submission_manifest.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True);p.add_argument('--box-source',default='tracks_2d_by_camera');a=p.parse_args()
    print(json.dumps(export(a.metadata,a.predictions,a.output,a.box_source),indent=2))
