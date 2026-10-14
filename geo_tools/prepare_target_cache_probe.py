from common import *


def main():
    stages=[]
    specs=[('cache_vggt_prefix40_scene1_v1','vggt',40),('cache_vggt_prefix60_scene1_v1','vggt',60),
        ('cache_vggt_fixed20_total80_scene1_full_v1','vggt',None),('cache_gmt_fixed20_scene1_full_v1','gmt',None),('cache_dino_fixed20_total40_scene1_full_v1','dinov2',None)]
    for tag,backbone,limit in specs:
        head=RUNTIME/'heads/low_rank_three_backbones_train_v1'/f'{backbone}_seed20261014.pt'
        cost=2 if backbone=='vggt' else 1
        command=['geo_tools/run_online.py','--video-id','1','--tag',tag,'--backbone',backbone,'--feature-weight','2','--max-residual','2',
            '--layer','17','--projection-dim','256' if backbone=='dinov2' else '0','--pooling','reid' if backbone=='gmt' else 'roi_align',
            '--head-path',str(head),'--policy','fixed','--interval','20','--min-interval','1','--call-budget',str(10*cost),'--total-call-budget',str(40*cost),'--reuse-target-features']
        if limit:command+=['--max-frames',str(limit)]
        stages.append(dict(id=tag,cpu_only=False,command=command))
        if limit is None:
            stages.append(dict(id='eval_'+tag,cpu_only=True,depends_on=[tag],command=['geo_tools/evaluate_predictions.py','--prediction',str(RUNTIME/'online'/tag/'predictions.json'),
                '--video-id','1','--run-manifest',str(RUNTIME/'online'/tag/'manifest.json')]))
    save_json(ROOT/'manifests/target_cache_probe_queue_v1.json',dict(id='target_cache_probe_v1',project_gpu_ids=['4'],queues=dict(gpu4=stages),
        scope='prospective causal scene1 bounded feature-reuse probe; actual current read/detection/GTA and encoder costs charged; no GT/TEST tuning',
        comparison_scope='40 successful feature-observation updates per backbone; actual encoder calls differ (VGGT two-order80, DINO40, native GMT0); no claim of identical encoder budgets across backbones'))


if __name__=='__main__':main()
