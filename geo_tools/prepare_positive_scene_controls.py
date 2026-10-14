"""Prespecified independent controls for the complete causal scene1 signal."""
from common import *


def main():
    controls=[
        ('C_gmt_head_dense_causal_v1_v1','gmt',str(RUNTIME/'heads/low_rank_three_backbones_train_v1/gmt_seed20261014.pt'),0,'reid',0.),
        ('C_dino_head_dense_causal_v1_v1','dinov2',str(RUNTIME/'heads/low_rank_three_backbones_train_v1/dinov2_seed20261014.pt'),256,'roi_align',0.),
        ('B5_vggt_head_dense_causal_v1_v1','vggt',str(RUNTIME/'heads/low_rank_three_backbones_train_v1/vggt_seed20261014.pt'),0,'roi_align',0.),
        ('B6_vggt_camera_dense_causal_v1_v1','vggt','',0,'roi_align',.5),
    ];stages=[]
    for tag,backbone,head,projection,pool,geometry in controls:
        command=['geo_tools/run_online.py','--video-id','1','--tag',tag,'--backbone',backbone,'--feature-weight','0' if geometry else '2',
            '--max-residual','2','--layer','17','--projection-dim',str(projection),'--pooling',pool,'--geometry-weight',str(geometry),'--geometry-method','vggt_camera',
            '--policy','fixed','--interval','1','--call-budget','200','--min-interval','1']
        if head:command+=['--head-path',head]
        stages.append(dict(id=tag,command=command,cpu_only=False))
        stages.append(dict(id='eval_'+tag,cpu_only=True,depends_on=[tag],command=['geo_tools/evaluate_predictions.py','--prediction',str(RUNTIME/'online'/tag/'predictions.json'),
            '--video-id','1','--run-manifest',str(RUNTIME/'online'/tag/'manifest.json')]))
    save_json(ROOT/'manifests/positive_scene_controls_v1.json',dict(id='positive_scene_controls_v1',project_gpu_ids=['4'],queues=dict(gpu4=stages),
        scope='full causal scene1 independent controls after first complete B7 dense signal; fixed TRAIN seed14 settings; no TEST selection',
        limitations='masked geometry-only input differs from the unmasked shared feature+pose input; feature-only control directly isolates the geometry residual effect under otherwise same features'))


if __name__=='__main__':main()
