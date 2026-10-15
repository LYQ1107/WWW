"""One real old V2 target checks the new pool helper bit-for-bit."""
import argparse
import numpy as np
import torch
from common import *
from trackvggt_token_pool import full_image_target_pool


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['vggt','dinov2'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    old=Path('/data1/liuyeqiang/WWW_vggt_geo_runtime');new=Path('/data1/liuyeqiang/WWW_trackvggt_runtime')
    manifest_path=new/'tokens'/f'{a.backbone}_scene4_full_v2'/'manifest.json';manifest=json.loads(manifest_path.read_text());row=manifest['rows'][0]
    if sha256(row['file'])!=row['sha256']:raise ValueError('old target changed')
    payload=torch.load(row['event_file'],map_location='cpu');event=payload['events'][row['event_index']];query=payload['observations'][tuple(event['query_source'])]
    data=train_data();images=sorted([r for r in video_records(data,4) if r['frame_id']==query['frame']],key=lambda r:r['view_id'])
    if [r['image_id'] for r in images]!=row['input_image_ids']:raise ValueError('physical views changed')
    sys.path[:0]=[str(old/'sources/vggt'),str(old/'sources/dinov2')]
    from vggt.utils.load_fn import load_and_preprocess_images
    from gtr.geometry.trackvggt_features import FrozenTrackVGGT
    from gtr.geometry.track_dinov2_features import FrozenTrackDINO
    checkpoint=Path(manifest['backbone_checkpoint'])
    if sha256(checkpoint)!=manifest['backbone_checkpoint_sha256']:raise ValueError('frozen weights changed')
    seed_all(20261014);encoder=(FrozenTrackVGGT if a.backbone=='vggt' else FrozenTrackDINO)(checkpoint)
    layers,attention,_=encoder.extract(load_and_preprocess_images([r['file_name'] for r in images],mode='pad')[None])
    view=next(i for i,r in enumerate(images) if r['view_id']==query['camera']);arrays,counts=full_image_target_pool(layers,attention,view,query)
    comparisons={}
    with np.load(row['file']) as reference:
        if set(reference.files)!=set(arrays):raise ValueError('array interface changed')
        for name,value in arrays.items():
            comparisons[name]=dict(exact=bool(np.array_equal(value,reference[name])),shape=list(value.shape))
    if not all(v['exact'] for v in comparisons.values()):raise ValueError('new target pool differs from completed V2 target')
    save_json(a.output,dict(status='PASS_REAL_V2_POOL_BITWISE_PARITY',backbone=a.backbone,key=row['key'],old_manifest=str(manifest_path),old_manifest_sha256=sha256(manifest_path),old_target_sha256=row['sha256'],
        arrays=comparisons,counts=counts,encoder_calls=encoder.calls,source_helper_sha256=sha256(ROOT/'geo_tools/trackvggt_token_pool.py'),old_target_unchanged=sha256(row['file'])==row['sha256'],
        scope='Single real interface-overlap gate for new exporter, not teacher efficacy or full-scope encoder equality',provenance=provenance()))
    print(json.dumps(dict(status='PASS_REAL_V2_POOL_BITWISE_PARITY',backbone=a.backbone,key=row['key'])),flush=True)


if __name__=='__main__':main()
