#!/usr/bin/env python3
"""Continuous clip PWQT training with explicit TBPTT and full coverage logs."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.training.clip_dataset import WorldQueryClipDataset,collate_clips
from worldmot.training.train_utils import save_checkpoint,seed_everything,restore_rng_state
from worldmot.training.world_query_criterion import TemporalQueryCriterion
from experiment_provenance import source_stamp,cache_stamp

COUNT_FIELDS=('positive_observations','NULL_observations','birth_targets','ambiguous_masked','legal_positive_observations','FP_observations')


def train_step(model,row,criterion,tbptt,optimizer=None,max_missed=64):
    b,T,N,_=row['observations'].shape;state=model.init_state(b,row['observations'].device)
    logs=[];segment=[];grad_norms=[];real_frames=0
    identity_cpu=row['identity'].detach().cpu().numpy()
    valid_cpu=row['observation_valid'].detach().cpu().numpy()
    time_cpu=row['time_valid'].detach().cpu().numpy()
    for t in range(T):
        output=model(row['observations'][:,t],row['geometry'][:,t],state,row['observation_valid'][:,t],row['camera_ids'][:,t])
        identity=row['identity'][:,t];valid=row['observation_valid'][:,t];time_valid=row['time_valid'][:,t]
        losses=criterion(output,identity,valid,row['camera_ids'][:,t],state.global_ids,time_valid)
        segment.append(losses['total']);logs.append({k:float(v.detach()) for k,v in losses.items()})
        mappings=[];born=[]
        for bi in range(b):
            mapping={};new={}
            if bool(time_cpu[bi,t]):
                gids=state.global_ids[bi].tolist();slot_by_id={gid:q for q,gid in enumerate(gids) if gid>=0};free=[q for q,gid in enumerate(gids) if gid<0]
                for o in range(N):
                    gid=int(identity_cpu[bi,t,o])
                    if gid<0 or not bool(valid_cpu[bi,t,o]):continue
                    if gid not in slot_by_id:
                        if not free:raise RuntimeError('training clip identity capacity exceeded')
                        q=free.pop(0);slot_by_id[gid]=q;new[q]=gid
                    mapping.setdefault(slot_by_id[gid],[]).append(o)
            mappings.append(mapping);born.append(new)
        state=model.accept_observations(output['state'],output,mappings,row['world_xyz'][:,t],row['world_covariance'][:,t],row['geometry_valid'][:,t])
        for bi in range(b):
            for q,gid in born[bi].items():state.global_ids[bi,q]=gid
        state.global_ids = state.global_ids.masked_fill(state.missed > max_missed, -1)
        real_frames+=int(time_valid.sum())
        if (t+1)%tbptt==0 or t==T-1:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);torch.stack(segment).mean().backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                if not torch.isfinite(norm):raise RuntimeError('nonfinite training gradient')
                optimizer.step();grad_norms.append(float(norm))
            segment=[];state=state.detach()
    avg={k:float(np.mean([r[k] for r in logs])) for k in logs[0]}
    for key in COUNT_FIELDS:avg[key]=int(sum(r[key] for r in logs))
    avg['gradient_norm']=float(np.mean(grad_norms)) if grad_norms else 0.;avg['real_frames']=real_frames
    return avg


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--cache',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--seed',type=int,default=20261022);p.add_argument('--device',default='cuda:0');p.add_argument('--scenes',nargs='+',type=int)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--tbptt',type=int,default=8);p.add_argument('--clip-lengths',nargs='+',type=int,default=[8,16,32])
    p.add_argument('--hidden-dim',type=int,default=128);p.add_argument('--max-queries',type=int,default=128);p.add_argument('--ablation',choices=['joint','no_omega','no_geometry','no_appearance','two_stage'],default='joint')
    p.add_argument('--resume');p.add_argument('--lr',type=float,default=3e-4);a=p.parse_args(argv)
    torch.set_num_threads(2);seed_everything(a.seed);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    ds=WorldQueryClipDataset(a.cache,scenes=a.scenes,clip_length=a.clip_lengths[0])
    if not len(ds):raise RuntimeError('empty feature cache')
    model=PersistentWorldQueryTransformer(observation_dim=532,hidden_dim=a.hidden_dim,max_world_queries=a.max_queries,ablation=a.ablation).to(a.device)
    provenance=source_stamp();cache_provenance=cache_stamp(a.cache,ds.scenes)
    cache_schemas=sorted({m['schema'] for m in cache_provenance.values()})
    gmt_frontend=any('_gmt_' in schema for schema in cache_schemas)
    criterion=TemporalQueryCriterion();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);start_epoch=0;history=[]
    if a.resume:
        payload=torch.load(a.resume,map_location=a.device,weights_only=True)
        if payload['manifest'].get('source_provenance')!=provenance or payload['manifest'].get('cache_provenance')!=cache_provenance:
            raise RuntimeError('resume source/cache differs from checkpoint; preserve prior trial before a fresh run')
        if 'rng_state' not in payload:raise RuntimeError('exact resume requires saved RNG state')
        model.load_state_dict(payload['model']);opt.load_state_dict(payload['optimizer']);start_epoch=payload['epoch'];history=payload['manifest']['history'];restore_rng_state(payload['rng_state'])
    manifest={'status':'FULLTRAIN_RUNNING','seed':a.seed,'model_config':model.config,'cache':a.cache,'train_scenes':sorted(ds.scenes),'epochs_requested':a.epochs,
              'frames_per_scene':{str(s):len(f) for s,f in ds.scenes.items()},'total_timestamps':sum(len(f) for f in ds.scenes.values()),'tbptt':a.tbptt,'clip_curriculum':a.clip_lengths,
              'initialization':'fixed_seed_random_pwqt_frozen_official_rtmdet_omega','history':history,'formal_inference_allowed':False,'optimizer':'AdamW','learning_rate':a.lr,'max_missed':64,
              'gradient_clip_norm':1.,'gradient_norm_log':'pre_clip','supervision_counts':'actual epoch sums; losses are batch/step means'}
    manifest.update(source_provenance=provenance,cache_provenance=cache_provenance,
                    resume_checkpoint=str(a.resume) if a.resume else None,resume_rng_restored=bool(a.resume))
    manifest['feature_cache_schemas']=cache_schemas
    manifest['detector_frontend']='gmt_pre_gta_detector' if gmt_frontend else 'rtmdet'
    if gmt_frontend:manifest['initialization']='fixed_seed_random_pwqt_frozen_official_GMT_pre_GTA_omega'
    official_lengths = [932,1886,1200,901,1769,601,443,523,1942,2092,2057,1200,1621,969,1048,1034,1200,1029,1052,1001,1001,1000,1052,1201]
    complete_all_train = sorted(ds.scenes) == list(range(1,25)) and all(len(ds.scenes[s]) == official_lengths[s-1] for s in ds.scenes)
    manifest['full_train_coverage'] = bool(complete_all_train)
    manifest['training_scope'] = 'FULL_24_TRAIN' if complete_all_train else 'TRAIN_SUBSET_OR_REGRESSION'
    for epoch in range(start_epoch,a.epochs):
        # A fixed prefix makes a 4-epoch refit match the first four epochs of
        # the 6-epoch internal trial. All candidates include length 32.
        length=a.clip_lengths[min(len(a.clip_lengths)-1,epoch)];ds.set_clip_length(length)
        # Preserve full coverage while reducing padding of crowded and sparse
        # clips in the same batch.  Bucket order is seed/epoch randomized.
        ordered=sorted(range(len(ds)),key=lambda i:max((len(f['observations']) for f in ds.scenes[ds.clips[i][0]][ds.clips[i][1]:ds.clips[i][2]]),default=0))
        batches=[ordered[i:i+a.batch_size] for i in range(0,len(ordered),a.batch_size)]
        np.random.default_rng(a.seed+epoch).shuffle(batches)
        loader=DataLoader(ds,batch_sampler=batches,collate_fn=collate_clips,num_workers=0)
        model.train();records=[];coverage={s:0 for s in ds.scenes};started=time.time()
        for i,row in enumerate(loader):
            for s,ts in zip(row['scene'].tolist(),row['time_valid']):coverage[s]+=int(ts.sum())
            row={k:v.to(a.device) for k,v in row.items()};r=train_step(model,row,criterion,a.tbptt,opt);records.append(r)
            if i%25==0:print(json.dumps({'epoch':epoch+1,'batch':i,'batches':len(loader),'loss':r['total'],'grad':r['gradient_norm'],'elapsed_s':round(time.time()-started,1)}),flush=True)
        if coverage!={s:len(f) for s,f in ds.scenes.items()}:raise RuntimeError('incomplete epoch coverage')
        metric={k:float(np.mean([r[k] for r in records])) for k in records[0]};metric.update(epoch=epoch+1,clip_length=length,coverage=coverage,elapsed_s=time.time()-started)
        for key in COUNT_FIELDS:metric[key]=sum(r[key] for r in records)
        metric['real_frames']=sum(r['real_frames'] for r in records)
        metric['positive_to_FP_ratio']=metric['legal_positive_observations']/metric['FP_observations'] if metric['FP_observations'] else None
        history.append(metric);manifest['history']=history;manifest['epochs_completed']=epoch+1
        manifest['status']=('FULLTRAIN_COMPLETE' if complete_all_train else 'TRAIN_SUBSET_COMPLETE') if epoch+1==a.epochs else 'TRAINING_RUNNING'
        save_checkpoint(out.with_suffix('.tmp.pt'),model,opt,epoch+1,manifest);os.replace(out.with_suffix('.tmp.pt'),out)
        if epoch+1 in (2,4,6):
            save_checkpoint(out.with_name(out.stem+f'.epoch{epoch+1}.pt'),model,opt,epoch+1,manifest)
        out.with_suffix('.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(metric),flush=True)
    print(json.dumps({'status':manifest['status'],'checkpoint':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'timestamps':manifest['total_timestamps']}),flush=True)
if __name__=='__main__':main()
