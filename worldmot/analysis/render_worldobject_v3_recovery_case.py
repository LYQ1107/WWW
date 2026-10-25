#!/usr/bin/env python3
"""Render an actual TRAIN-only recovery/new-ID failure with original RGB."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from summarize_worldobject_v3 import read,sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    folder=a.root/'diagnostics/retained_TRAIN_VAL/FULL_WORLD_OBJECT_V3/epoch_01'
    report=read(folder/'attribution.json')
    cases=[e for s in report['per_scene'].values() for e in s['analysis']['examples'].get('RECOVERY_NEXT_FRAME_ID_BREAK',[])
        if e['new_birth_indices'] and e['geometry_rejected_indices']]
    case=min(cases,key=lambda x:(x['scene'],x['timestamp']));sid=case['scene'];time=case['timestamp'];label=case['label']
    source=folder/f'scene_{sid:02d}/tracks.jsonl';manifest=read(source.with_name('manifest.json'))
    if sha(source)!=manifest['tracks_sha256']:raise ValueError('IMMUTABLE_TRAIN_PREDICTION_CHANGED')
    times=[time-2,time-1,time];outputs={}
    for line in source.open():
        row=json.loads(line)
        if row['timestamp'] in times:outputs[row['timestamp']]=row
        if row['timestamp']==time:break
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.training.world_object_identity_targets import match_train_identities
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack',split='train',load_annotations=True)
    pairs={x.frame_id:x for x in ds.synchronized_pairs(sid,start_frame=times[0],limit=3)}
    selected={};image_proofs=[]
    for t in times:
        row=outputs[t];boxes=[x for items in row['tracks_2d_by_camera'].values() for x in items]
        obs=[SimpleNamespace(key=(x['camera_id'],x['detection_id']),camera_id=x['camera_id'],bbox_xywh=np.asarray(x['bbox_xywh'])) for x in boxes]
        labels,_=match_train_identities(obs,ds.ground_truth_pair(pairs[t]))
        selected[t]={x['camera_id']:x for x,gt in zip(boxes,labels) if gt==label}
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for view in (1,2):
        bounds=np.array([selected[t][view]['bbox_xywh'] for t in times if view in selected[t]])
        lo=bounds[:,:2].min(0);hi=(bounds[:,:2]+bounds[:,2:]).max(0);pad=.45*np.maximum(hi-lo,20)
        for col,t in enumerate(times):
            axis=axes[view-1,col];path=ds.image_path(pairs[t].images[view]);rgb=np.asarray(Image.open(path).convert('RGB'))
            image_proofs.append({'timestamp':t,'camera':view,'RGB_sha256':sha(path)})
            axis.imshow(rgb);det=selected[t].get(view)
            if det:
                x,y,w,h=det['bbox_xywh'];gid=det['global_id'];color=plt.cm.tab20(gid%20)
                axis.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,linewidth=3))
                axis.text(x,y-3,f'GT {label} / delivered ID {gid}',fontsize=11,color='white',
                    bbox={'facecolor':color,'edgecolor':'none','alpha':.95})
            axis.set_xlim(max(0,lo[0]-pad[0]),min(rgb.shape[1],hi[0]+pad[0]))
            axis.set_ylim(min(rgb.shape[0],hi[1]+pad[1]),max(0,lo[1]-pad[1]))
            decision=next((x['decision'] for x in outputs[t]['decisions'] if det and x['camera_id']==view and x['detection_id']==det['detection_id']),None)
            suffix=' / visual recovery' if decision=='VISUAL_RECOVERY_SPATIAL_CONFLICT' else ' / next frame' if t==time else ''
            axis.set_title(f'Camera {view} / t={t}'+suffix)
            axis.axis('off')
    fig.suptitle(f'Actual TRAIN VAL Scene {sid}: same clean GT person, recovered ID changes next frame',fontsize=13)
    fig.text(.025,.045,'TRAIN diagnostic only. Display crops; model inputs were unchanged. Original GT/outputs are not edited.',fontsize=9)
    fig.text(.025,.022,'R2 selection-epoch-1 weights d4644ff9...; not the final TEST checkpoint. Scene 24 is excluded from official raw-GT scoring.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.96));a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'TRAIN_scene24_recovery_next_ID.png';fig.savefig(path,dpi=135);plt.close(fig)
    write_json(a.output/'manifest.json',{'schema':'worldobject_v3_actual_TRAIN_recovery_visual_v1',
        'image':path.name,'image_sha256':sha(path),'prediction_sha256':manifest['tracks_sha256'],
        'attribution_report_sha256':sha(folder/'attribution.json'),'checkpoint_sha256':report['checkpoint_sha256'],
        'selection':'Earliest actual TRAIN clean recovery next-frame new-ID event with direct geometry rejection',
        'source_event':case,'RGB_sources':image_proofs,'renderer_sha256':sha(__file__),'model_inputs_cropped':False,'display_crops_only':True,
        'GT_role':'POSTHOC_CLEAN_TRAIN_LABELS','official_TEST_result':False})
    print(json.dumps({'image':str(path),'bytes':path.stat().st_size,'event_time':time}))


if __name__=='__main__':main()
