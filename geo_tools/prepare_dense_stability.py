"""Finite prospective dense three-head-seed and camera-repair verification."""
from common import *


def main():
    strict=[1,10,11,18,19]
    queues={'gpu0':[],'gpu5':[]}
    def stage(queue,tag,command,cpu=False,depends=()):
        queues[queue].append(dict(id=tag,command=command,cpu_only=cpu,depends_on=list(depends)))
    def online(queue,video,tag,options):
        stage(queue,tag,['geo_tools/run_online.py','--video-id',str(video),'--tag',tag,*options])
        eval_tag='eval_'+tag
        stage(queue,eval_tag,['geo_tools/evaluate_predictions.py','--prediction',str(RUNTIME/'online'/tag/'predictions.json'),
                             '--video-id',str(video),'--run-manifest',str(RUNTIME/'online'/tag/'manifest.json')],True,[tag])
        return RUNTIME/'online'/tag/'predictions.json',eval_tag
    def suite(queue,tag,paths,dependencies):
        source=ROOT/'manifests'/f'{tag}_sources.json'
        save_json(source,dict(video_predictions={str(v):str(p) for v,p in paths.items()},all_frames=True,
                              protocol='complete causal strict5, fixed inference RNG, declared independent TRAIN head seed; no TEST'))
        stage(queue,'combine_'+tag,['geo_tools/evaluate_suite.py','--sources',str(source.relative_to(ROOT)),'--tag',tag,
                                   '--report-output',str(RUNTIME/'matrix_reports'/f'{tag}_metrics.json')],True,dependencies)
    # Scene1 actual corrected-camera control already finished; never recompute
    # it under a new provenance or replace the locked original baseline.
    camera={1:RUNTIME/'online/camera_index_corrected_B0_scene1_full_v1/predictions.json'};deps=[]
    for video in strict[1:]:
        tag=f'camera_index_corrected_B0_strict_suite_v1_v{video}'
        camera[video],dependency=online('gpu0',video,tag,['--validate-gmt-camera-index'])
        deps.append(dependency)
    suite('gpu0','camera_index_corrected_B0_strict_suite_v1',camera,deps)
    # Feature-only seed14 is already complete for scene1. Remaining scenes
    # isolate actual geometry contribution; then evaluate all three TRAIN head
    # seeds at the same dense settings. Parameters are fixed before outcomes.
    for geometry in [0.,.5]:
        for seed in [20261014,20261015,20261016]:
            if geometry and seed==20261014:
                continue  # Original immutable dense B7 queue evaluates this.
            prefix=f'causal_B{7 if geometry else 5}_dense_seed{seed}_strict_suite_v1'
            paths={};dependencies=[]
            for video in strict:
                if not geometry and seed==20261014 and video==1:
                    paths[video]=RUNTIME/'online/B5_vggt_head_dense_causal_v1_v1/predictions.json'
                    continue
                tag=f'{prefix}_v{video}'
                options=['--feature-weight','2','--max-residual','2','--layer','17','--projection-dim','0',
                         '--head-path',str(RUNTIME/'heads/low_rank_three_backbones_train_v1'/f'vggt_seed{seed}.pt'),
                         '--geometry-weight',str(geometry),'--geometry-method','vggt_camera',
                         '--policy','fixed','--interval','1','--min-interval','1','--call-budget','200']
                paths[video],dependency=online('gpu5',video,tag,options)
                dependencies.append(dependency)
            suite('gpu5',prefix,paths,dependencies)
    save_json(ROOT/'manifests/dense_stability_and_camera_queue_v1.json',
              dict(id='dense_stability_and_camera_v1',project_gpu_ids=['0','5'],queues=queues,
                   scope='fixed dense feature-only ablation and head-seed stability, plus explicit corrected-camera GMT baseline control; all strict development scenes',
                   inference_seed=20261014,head_training_seeds=[20261014,20261015,20261016],
                   used_test_for_selection=False,parameter_search=False,
                   completion_not_implied_by_queue_exit=True,
                   limitations='Dense stability does not automatically establish a significant gain; camera correction is a separate validation flag, not a new production baseline. No cached throughput used.'))
    print(json.dumps({queue:len(stages) for queue,stages in queues.items()}))


if __name__=='__main__':main()
