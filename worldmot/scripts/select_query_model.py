#!/usr/bin/env python3
"""Model/threshold selection exclusively on the held-out TRAIN scenes."""
from __future__ import annotations
import argparse,copy,csv,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.training.clip_dataset import WorldQueryClipDataset,collate_clips
from worldmot.training.world_query_criterion import TemporalQueryCriterion
from worldmot.tracking.persistent_query_runtime import PersistentQueryRuntime
from train_world_query import train_step
from replay_query_cache import tokens_from_cache

VAL_SCENES=[4,8,12,16,20,24]

def validation_loss(ds,checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model=PersistentWorldQueryTransformer(**saved['manifest']['model_config']).to(device);model.load_state_dict(saved['model']);model.eval()
    criterion=TemporalQueryCriterion();values=[]
    with torch.no_grad():
        for row in DataLoader(ds,batch_size=1,collate_fn=collate_clips):
            row={k:v.to(device) for k,v in row.items()};values.append(train_step(model,row,criterion,8))
    return {k:float(np.mean([r[k] for r in values])) for k in values[0]}

def calibration(ds,checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);samples=[]
    # Fixed 32-frame start/middle/end windows in every TRAIN-validation scene.
    # This is a TRAIN-only selection surrogate, never a formal MOT score.
    for sid,frames in sorted(ds.scenes.items()):
        for start in sorted({0,max(0,len(frames)//2-16),max(0,len(frames)-32)}):samples.append((sid,start,frames[start:start+32]))
    results=[]
    for margin,birth,pair in itertools.product([0.,.5,1.],[.3,.5,.7],[.5,.7]):
        model=PersistentWorldQueryTransformer(**saved['manifest']['model_config'])
        runtime=PersistentQueryRuntime(model,device,checkpoint,null_margin=margin,birth_threshold=birth,pair_threshold=pair)
        total_gt=total_predictions=total_correct=births=assigned=0;clip_records=[]
        for sid,start,frames in samples:
            runtime.reset(sid);gc={};pc={};counts={};clip_assigned=0;clip_births=0
            for frame in frames:
                n=len(frame['observations']);data={**frame,'frame_offsets':np.asarray([0,n])}
                tokens=tokens_from_cache(data,0,frame['timestamp']);r=runtime.step(tokens,frame['timestamp']);clip_assigned+=len(r.global_ids_by_observation);clip_births+=len(r.births)
                for o,identity in enumerate(frame['identity']):
                    identity=int(identity)
                    if identity==-2:continue
                    if identity>=0:gc[identity]=gc.get(identity,0)+1
                    gid=r.global_ids_by_observation.get(o)
                    if gid is not None:
                        pc[gid]=pc.get(gid,0)+1
                        if identity>=0:counts[(identity,gid)]=counts.get((identity,gid),0)+1
            keys_g=sorted(gc);keys_p=sorted(pc);matrix=np.zeros((len(keys_g),len(keys_p)))
            for (g,p),n in counts.items():matrix[keys_g.index(g),keys_p.index(p)]=n
            correct=0
            if matrix.size:
                rr,cc=linear_sum_assignment(-matrix);correct=int(matrix[rr,cc].sum())
            total_correct+=correct;total_gt+=sum(gc.values());total_predictions+=sum(pc.values());assigned+=clip_assigned;births+=clip_births
            clip_records.append({'scene':sid,'start':start,'frames':len(frames),'assigned':clip_assigned,'births':clip_births})
        f1=2*total_correct/max(1,total_gt+total_predictions)
        row={'null_margin':margin,'birth_threshold':birth,'pair_threshold':pair,'identity_f1_surrogate':f1,'IDTP_surrogate':total_correct,'gt_cached_positive_observations':total_gt,'predictions':total_predictions,'births':births,'assigned':assigned,'clips':clip_records}
        results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='clips'}),flush=True)
    # Prefer the higher confidence threshold on exact ties; never TEST data.
    best=max(results,key=lambda r:(r['identity_f1_surrogate'],r['birth_threshold'],r['pair_threshold'],r['null_margin']))
    return results,best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--internal-root',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);ds=WorldQueryClipDataset([a.cache],clip_length=32,scenes=VAL_SCENES)
    if set(ds.scenes)!=set(VAL_SCENES):raise RuntimeError('all six held-out TRAIN scenes are required')
    losses=[]
    for epoch in [2,4,6]:
        checkpoint=Path(a.internal_root)/f'joint.epoch{epoch}.pt';values=validation_loss(ds,checkpoint,a.device)
        losses.append({'epoch':epoch,'checkpoint':str(checkpoint),**values});print(json.dumps(losses[-1]),flush=True)
    # Final refit must contain all 8/16/32 curriculum lengths. Epoch 2 is a
    # diagnostic curve point; only epoch budgets 4 and 6 are eligible.
    selected_epoch=min([r for r in losses if r['epoch']>=4],key=lambda r:r['total'])['epoch']
    final={'status':'TRAIN_ONLY_SELECTION','validation_scenes':VAL_SCENES,'loss_evaluation':'all validation timestamps in continuous 32-frame clips','threshold_evaluation':'fixed start/middle/end 32-frame windows; cached identity-F1 selection surrogate, not HOTA or official IDF1','selected_epochs':selected_epoch,'epoch_validation':losses,'methods':{}}
    for method in ['joint','no_omega','no_appearance','two_stage','no_geometry']:
        cp=Path(a.internal_root)/f'{method}.epoch{selected_epoch}.pt';grid,best=calibration(ds,cp,a.device)
        final['methods'][method]={'thresholds':{k:best[k] for k in ['null_margin','birth_threshold','pair_threshold']},'validation_identity_f1_surrogate':best['identity_f1_surrogate'],'candidates':grid}
        (out/'selection.json').write_text(json.dumps(final,indent=2))
    final['status']='FROZEN_FROM_TRAIN_VALIDATION';(out/'selection.json').write_text(json.dumps(final,indent=2))
