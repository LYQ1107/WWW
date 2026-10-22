#!/usr/bin/env python3
"""Replay TRAIN feature observations; query timing is never E2E FPS."""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import torch
from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.models.world_observation_encoder import WorldObservationToken
from worldmot.tracking.persistent_query_runtime import PersistentQueryRuntime

def tokens_from_cache(data,i,timestamp):
    lo,hi=map(int,data['frame_offsets'][i:i+2]);tokens=[]
    for o in range(lo,hi):
        obs=data['observations'][o].astype(np.float32)
        tokens.append(WorldObservationToken(camera_id=int(data['camera_ids'][o]),timestamp=int(timestamp),detection_id=o-lo,
            image_feature=obs[:512],omega_feature=obs[256:512],xyz_world=data['xyz_world'][o],covariance=data['covariance'][o],
            bbox=data['bbox'][o],mask_feature=np.empty(0,np.float32),detection_score=float(obs[-2]),geometry_reliability=float(obs[-1]),
            geometry_feature=data['geometry'][o],valid=bool(data['observation_valid'][o]),source='frozen_rgb_rtmdet_omega_cache_v3',
            diagnostics={'geometry_coordinate_valid':bool(data['geometry_valid'][o]),'pure_3d_bbox_xywh':data['pure_projection_bbox'][o].tolist()}))
    return tokens

def replay(cache,checkpoint,output,device='cpu',scenes=None,null_margin=None,birth_threshold=None,pair_threshold=None):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model=PersistentWorldQueryTransformer(**saved['manifest']['model_config'])
    runtime=PersistentQueryRuntime(model,device=device,checkpoint=checkpoint,null_margin=null_margin,birth_threshold=birth_threshold,pair_threshold=pair_threshold)
    root=Path(output);root.mkdir(parents=True,exist_ok=True);summaries=[]
    for scene in sorted(Path(cache).glob('scene_*')):
        sid=int(scene.name.split('_')[-1])
        if scenes and sid not in scenes:continue
        runtime.reset(sid);out=root/scene.name;out.mkdir(exist_ok=True);frames=obs=assigned=births=0;lifetimes=collections.Counter();query_times=[]
        with (out/'tracks.jsonl').open('w') as f:
            for shard in sorted(scene.glob('shard_*.npz')):
                with np.load(shard,allow_pickle=False) as data:
                    metadata=json.loads(shard.with_suffix('.json').read_text())['per_frame']
                    for i,timestamp in enumerate(data['timestamps']):
                        tokens=tokens_from_cache(data,i,timestamp);t0=time.perf_counter();r=runtime.step(tokens,int(timestamp))
                        if device.startswith('cuda'):torch.cuda.synchronize(device)
                        query_times.append(time.perf_counter()-t0);frames+=1;obs+=len(tokens);assigned+=len(r.global_ids_by_observation);births+=len(r.births)
                        boxes={'1':[],'2':[]};pure={'1':[],'2':[]}
                        for o,gid in r.global_ids_by_observation.items():
                            token=tokens[o];v=str(token.camera_id);lifetimes[gid]+=1
                            boxes[v].append({'global_id':gid,'bbox_xywh':token.bbox.tolist(),'visible':True,'evaluation_eligible':True,'output_source':'CURRENT_ASSIGNED_DETECTOR_BBOX','score':token.detection_score})
                            pure[v].append({'global_id':gid,'bbox_xywh':token.diagnostics['pure_3d_bbox_xywh'],'evaluation_eligible':False})
                        row={'scene_id':sid,'timestamp':int(timestamp),'tracks_2d_by_camera':boxes,'pure_3d_projection_by_camera':pure,
                             'query_runtime':r.diagnostics,'birth_global_ids':r.births,'image_ids':metadata[i]['image_ids'],'query_only_seconds':query_times[-1],
                             'causal':True,'cache_replay_not_e2e':True}
                        f.write(json.dumps(row)+'\n')
        summary={'scene':sid,'frames':frames,'observations':obs,'assigned':assigned,'births':births,'unique_ids':len(lifetimes),
                 'ids_with_multiple_observations':sum(n>1 for n in lifetimes.values()),'maximum_id_observations':max(lifetimes.values(),default=0),
                 'query_only_total_s':sum(query_times),'formal_e2e_fps':None,'checkpoint_scope':saved['manifest'].get('training_scope'),
                 'thresholds':{'null_margin':runtime.null_margin,'birth_threshold':runtime.birth_threshold,'pair_threshold':runtime.pair_threshold}}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));summaries.append(summary);print(json.dumps(summary),flush=True)
    (root/'summary.json').write_text(json.dumps(summaries,indent=2));return summaries

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--scenes',nargs='+',type=int)
    p.add_argument('--null-margin',type=float);p.add_argument('--birth-threshold',type=float);p.add_argument('--pair-threshold',type=float);a=p.parse_args();torch.set_num_threads(2)
    replay(a.cache,a.checkpoint,a.output,a.device,a.scenes,a.null_margin,a.birth_threshold,a.pair_threshold)
