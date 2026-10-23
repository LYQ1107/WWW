#!/usr/bin/env python3
"""Real official checkpoint preflight on synchronized TRAIN RGB images."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from worldmot.models.vggt_point_backend import VGGTPointBackend


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metadata',required=True)
    p.add_argument('--dataset-root',default='/data/DATASETS/TRACKING/JDE/VisionTrack')
    p.add_argument('--scene',type=int,default=8)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.scene not in [4,8,12,16,20,24]:raise ValueError('TRAIN validation only')
    meta=json.loads(Path(args.metadata).read_text())
    if meta.get('annotations'):raise ValueError('GT metadata in inference')
    rows=[r for r in meta['images'] if r['video_id']==args.scene]
    first=min(r['frame_id'] for r in rows)
    rows=sorted([r for r in rows if r['frame_id']==first],key=lambda r:r['view_id'])
    images=[np.asarray(Image.open(Path(args.dataset_root)/'train'/r['file_name']).convert('RGB')) for r in rows]
    import torch
    torch.set_num_threads(2)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    backend=VGGTPointBackend(device=args.device)
    from vggt.utils.load_fn import load_and_preprocess_images
    from worldmot.geometry.vggt_point_coordinates import VGGTPointCoordinates
    official=load_and_preprocess_images([str(Path(args.dataset_root)/'train'/r['file_name']) for r in rows],mode='pad').numpy()
    reproduced=np.stack([VGGTPointCoordinates.from_shape(im.shape).preprocess(im) for im in images])
    assert np.array_equal(official,reproduced),'Image transform differs from official pad preprocessing'
    report={'status':'MODEL_LOADED_FORWARD_RUNNING','backend':backend.manifest(),
            'hardware':torch.cuda.get_device_name(0),'scene':args.scene,'timestamp':first,
            'official_preprocessing_exact':True,'forwards':[]}
    out.write_text(json.dumps(report,indent=2))
    for direction in [0,1]:
        order=[direction,1-direction]
        im=images[direction];h,w=im.shape[:2]
        # Component smoke points are a spatial grid, not claimed person points.
        points=np.array([(x,y) for y in np.linspace(.2*h,.8*h,4) for x in np.linspace(.2*w,.8*w,4)],np.float32)
        keys=[(rows[i]['view_id'],first) for i in order]
        for repeat in range(2):
            tracks=backend.track([images[i] for i in order],points,image_keys=keys,
                reference_key=keys[0],current_timestamp=first)
            report['forwards'].append({'direction':direction,'repeat':repeat,'image_keys':keys,
                'xy':tracks.xy.tolist(),'visibility':tracks.visibility.tolist(),
                'confidence':tracks.confidence.tolist(),'valid':tracks.valid.tolist(),
                'timings':tracks.timings,'diagnostics':tracks.diagnostics})
            print(json.dumps({'direction':direction,'repeat':repeat,'timings':tracks.timings,
                              'diagnostics':tracks.diagnostics}),flush=True)
    report['status']='PASS_OFFICIAL_PRETRAINED_POINT_FORWARD_BOTH_ANCHORS'
    report['scope']='Component forward and coordinates; no person identity accuracy claim'
    out.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
