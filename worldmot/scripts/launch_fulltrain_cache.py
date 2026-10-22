#!/usr/bin/env python3
"""Launch four resumable scene workers; state and logs live with NAS assets."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from worldmot.data.visiontrack import VisionTrack

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/data1/liuyeqiang/nas/A100/WorldMOT/fulltrain_20261022');a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True);logs=root/'logs';logs.mkdir(exist_ok=True)
    ds=VisionTrack('/data/DATASETS/TRACKING/JDE/VisionTrack');groups=[[] for _ in range(4)];load=[0]*4
    for sid in sorted(ds.scene_ids(),key=lambda s:-len(ds.frames(s))):
        idx=load.index(min(load));groups[idx].append(sid);load[idx]+=len(ds.frames(sid))
    workers=[]
    for gpu,scenes in zip([0,2,3,4],groups):
        log=logs/f'cache_gpu{gpu}.log';cmd=[sys.executable,'scripts/cache_fulltrain.py','--output',str(root/'train_cache'),'--device',f'cuda:{gpu}','--scenes',*map(str,scenes)]
        with log.open('a') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})
        workers.append({'pid':proc.pid,'gpu':gpu,'scenes':scenes,'expected_frames':sum(len(ds.frames(s)) for s in scenes),'log':str(log),'command':cmd})
    (root/'cache_workers.json').write_text(json.dumps({'status':'RUNNING','workers':workers},indent=2));print(json.dumps(workers,indent=2))
