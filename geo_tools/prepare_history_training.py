"""Freeze a bounded training-only test of temporal identity ranking."""
from common import *


def main():
    stages=[];collections=[]
    for video in [5,8,12]:
        tag=f'history_train_keyframe_v1_v{video}';collections.append(tag)
        stages.append(dict(id=tag,command=['geo_tools/collect_history_train.py','--video-id',str(video),'--tag',tag],cpu_only=False,depends_on=[]))
    stages.append(dict(id='history_rank_three_backbones_train_v1',command=['geo_tools/train_history_heads.py','--tag','history_rank_three_backbones_train_v1'],
        cpu_only=False,depends_on=collections))
    save_json(ROOT/'manifests/history_training_queue_v1.json',dict(id='history_training_v1',project_gpu_ids=['4'],queues={'gpu4':stages},
        hypothesis='existing tiny same-time head objective can learn confidence against unmatched rather than historical candidate identity; train on actual past-candidate logits with uniform-offset-invariant ranking loss',
        frozen_protocol=dict(videos=[5,8,12],fit=[5,8],heldout_train=[12],max_frames=200,interval=5,max_history_age=20,
            steps=2000,seeds=[20261014,20261015,20261016],optimizer='AdamW lr.001 wd.0001, clip_grad10, scale clamp.01..10',
            warm_start='original same-seed/backbone TRAIN-only small heads; matched architecture/parameters, feature_weight2/max_residual2'),
        integrity='actual fresh GMT/VGGT/DINO target collection; live adapter returns original score objects; standalone OFF40 prefix equality before reading raw identity labels; sparse target vectors only, no dense token maps',
        selection='minimum heldout TRAIN12 loss at500/1000/1500/2000. Primary VGGT gate requires better heldout history Recall1 than native and newly trained GMT/DINO controls, more fixes than harms; no DEV/TEST fitting',
        gate_failure='report bounded negative TRAIN result; no expansion into DEV tuning',provenance=provenance()))


if __name__=='__main__':main()
