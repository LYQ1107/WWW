#!/usr/bin/env python3
"""GT-free complete TEST image availability, decoding and byte fingerprints."""
import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import time
from pathlib import Path
from PIL import Image


def check_image(item,image_root):
    row={'image_id':item['id'],'scene':item['video_id'],'view':item['view_id'],
        'timestamp':item['frame_id'],'file_name':item['file_name']}
    try:
        data=(image_root/item['file_name']).read_bytes()
        with Image.open(io.BytesIO(data)) as source:
            actual=source.size;source.convert('RGB').load()
        if actual!=(item['width'],item['height']):raise ValueError('image dimensions differ from metadata')
        row.update(status='RGB_DECODE_OK',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        row.update(status='RGB_DECODE_FAILED',error=type(error).__name__+': '+str(error))
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True)
    p.add_argument('--dataset-root',default='/data/DATASETS/TRACKING/JDE/VisionTrack')
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    metadata=Path(a.metadata);data=json.loads(metadata.read_text())
    if data.get('annotations'):raise ValueError('TEST availability audit accepts metadata only')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);images=Path(a.dataset_root)/'test'
    records=root/'test_RGB_image_checks.jsonl.gz';scenes=[];failures=[];started=time.perf_counter()
    with gzip.open(records,'wt') as stream,concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for video in sorted(data['videos'],key=lambda row:row['id']):
            rows=sorted([r for r in data['images'] if r['video_id']==video['id']],key=lambda r:(r['frame_id'],r['view_id']))
            good=0;total_bytes=0
            for result in pool.map(lambda r:check_image(r,images),rows):
                stream.write(json.dumps(result)+'\n')
                if result['status']=='RGB_DECODE_OK':good+=1;total_bytes+=result['bytes']
                else:failures.append(result)
            scenes.append({'scene':video['id'],'expected_images':len(rows),'decoded_images':good,
                'synchronized_timestamps':len({r['frame_id'] for r in rows}),'compressed_bytes':total_bytes})
            print(json.dumps(scenes[-1]),flush=True)
    report={'status':'ALL_TEST_RGB_DECODE_VERIFIED' if not failures else 'TEST_RGB_INCOMPLETE',
        'scenes':scenes,'expected_images':len(data['images']),'decoded_images':sum(s['decoded_images'] for s in scenes),
        'synchronized_timestamps':sum(s['synchronized_timestamps'] for s in scenes),
        'GT_read':False,'models_run':False,'hyperparameters_selected':False,
        'metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),'failures':failures,
        'per_image_fingerprints':str(records),'per_image_fingerprints_sha256':hashlib.sha256(records.read_bytes()).hexdigest(),
        'elapsed_s':time.perf_counter()-started,'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    temp=root/'test_RGB_decode_summary.json.tmp';temp.write_text(json.dumps(report,indent=2));os.replace(temp,root/'test_RGB_decode_summary.json')
    print(json.dumps({k:v for k,v in report.items() if k not in ['scenes','failures']},indent=2),flush=True)
    if failures:raise RuntimeError(str(len(failures))+' TEST images missing or undecodable; no frames may be dropped')
