#!/usr/bin/env python3
"""Render actual RGB, predicted IDs and actual A->B points without GT input."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from worldmot.data.visiontrack import VisionTrack


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-output',type=Path,required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with (a.scene_output/'tracks.jsonl').open() as f:row=json.loads(next(f))
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',annotation=a.metadata,split='train',load_annotations=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4.3));colors=plt.get_cmap('tab10')
    for ax,view in zip(axes,[1,2]):
        rec=ds.images[int(row['image_ids'][str(view)])]
        ax.imshow(Image.open(ds.image_path(rec)).convert('RGB'))
        for out in row['tracks_2d_by_camera'][str(view)]:
            x,y,w,h=out['bbox_xywh'];color=colors(out['global_id']%10)
            ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,linewidth=1.8))
            ax.text(x,y,f"predicted ID {out['global_id']}",color='white',fontsize=8,bbox={'facecolor':color,'alpha':.8})
        ax.set_title(f"Camera {view}: current GMT boxes + A3 IDs");ax.axis('off')
    predictions=[x for x in row['point_predictions'] if x['kind']=='SYNCHRONOUS_CROSS_VIEW' and x['reference_key'][0]==1]
    if not predictions:raise ValueError('Expected real first-frame A-to-B point prediction')
    pred=predictions[0];offset=0
    for detection_id,n in pred['source_detections']:
        src=next(o for o in row['tracks_2d_by_camera']['1'] if o['detection_id']==detection_id)
        sl=slice(offset,offset+n);offset+=n;color=colors(src['global_id']%10)
        query=np.asarray(pred['query_points'])[sl];target=np.asarray(pred['predicted_xy'])[1,sl]
        valid=np.asarray(pred['valid'])[1,sl].astype(bool)
        axes[0].scatter(query[:,0],query[:,1],s=10,c=[color],edgecolors='white',linewidths=.3)
        axes[1].scatter(target[valid,0],target[valid,1],s=20,c=[color],marker='x',linewidths=.9)
    confidence=np.asarray(pred['confidence'])[1];visibility=np.asarray(pred['visibility'])[1]
    fig.suptitle(f"TRAIN scene {row['scene_id']} timestamp {row['timestamp']} | circles: actual A query points; crosses: actual predictions in B\n"
        f"A-to-B median confidence {np.median(confidence):.3f}, visibility {np.median(visibility):.3f}; default confidence gate 0.500; no shared cross-view ID",fontsize=10)
    fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    fig.savefig(a.output,dpi=110,pil_kwargs={'quality':85});plt.close(fig)
    print(json.dumps({'output':str(a.output),'source_timestamp':row['timestamp'],'GT_used':False,'bytes':a.output.stat().st_size}))


if __name__=='__main__':main()
