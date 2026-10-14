"""Predeclare the remaining finite matrix; no metric-driven setting selection."""
from common import *


def main():
    queues={f'gpu{gpu}':[] for gpu in [0,4,5,6]}
    strict=[1,10,11,18,19]
    head=str(RUNTIME/'heads/low_rank_three_backbones_train_v1/vggt_seed20261014.pt')
    feature=['--feature-weight','2','--max-residual','2','--layer','17','--projection-dim','0','--head-path',head,'--geometry-weight','0.5','--geometry-method','vggt_camera']
    def stage(queue,identifier,command,cpu=False,depends=()):
        queues[queue].append(dict(id=identifier,command=command,cpu_only=cpu,depends_on=list(depends)))
    def online(queue,video,tag,policy,interval,total=-1,budget=200,offset=0,scheduler_seed=20261014):
        command=['geo_tools/run_online.py','--video-id',str(video),'--tag',tag,*feature,'--policy',policy,'--interval',str(interval),
                 '--call-budget',str(budget),'--total-call-budget',str(total),'--min-interval','1','--offset',str(offset),'--scheduler-seed',str(scheduler_seed)]
        stage(queue,tag,command)
        stage(queue,'eval_'+tag,['geo_tools/evaluate_predictions.py','--prediction',str(RUNTIME/'online'/tag/'predictions.json'),
              '--video-id',str(video),'--run-manifest',str(RUNTIME/'online'/tag/'manifest.json')],True,[tag])
        return RUNTIME/'online'/tag/'predictions.json'
    def suite(queue,tag,paths,dependencies):
        source=ROOT/'manifests'/f'{tag}_sources.json'
        save_json(source,dict(protocol='complete strict development; actual causal/offline protocol taken from each run manifest, never mixed',video_predictions={str(v):str(p) for v,p in paths.items()},all_frames=True))
        stage(queue,'combine_'+tag,['geo_tools/evaluate_suite.py','--sources',str(source.relative_to(ROOT)),'--tag',tag,
              '--report-output',str(RUNTIME/'matrix_reports'/f'{tag}_metrics.json')],True,dependencies)
    # The first scene18 budget comparison is already complete. Other scenes
    # use identical settings, and realized encoder counts are audited later.
    for queue,policy,interval,prefix,existing in [
        ('gpu0','fixed',20,'causal_B8_fixed20_total80_strict_suite_v1','online_B8_fixed20_total80_full_v18_v1'),
        ('gpu0','random',10,'causal_random_total80_strict_suite_v1','online_random_total80_full_v18_v1'),
        ('gpu4','joint',10,'causal_B9_joint_total80_strict_suite_v1','online_B9_joint_total80_full_v18_v1')]:
        paths={18:RUNTIME/'online'/existing/'predictions.json'};dependencies=[]
        for video in [1,10,11,19]:
            tag=f'{prefix}_v{video}'
            paths[video]=online(queue,video,tag,policy,interval,80,20)
            dependencies.append('eval_'+tag)
        suite(queue,prefix,paths,dependencies)
    for queue,interval in [('gpu5',1),('gpu5',5),('gpu6',10),('gpu6',20)]:
        prefix=f'causal_B7_fixed_k{interval}_strict_suite_v1'
        paths={};dependencies=[]
        for video in strict:
            tag=f'{prefix}_v{video}'
            paths[video]=online(queue,video,tag,'fixed',interval)
            dependencies.append('eval_'+tag)
        suite(queue,prefix,paths,dependencies)
    for policy in ['motion','identity']:
        prefix=f'causal_{policy}_total80_strict_suite_v1';paths={};dependencies=[]
        for video in strict:
            tag=f'{prefix}_v{video}'
            paths[video]=online('gpu6',video,tag,policy,10,80,20)
            dependencies.append('eval_'+tag)
        suite('gpu6',prefix,paths,dependencies)
    # Prefix offsets and RNG choices are fixed prospectively, not selected by
    # DEV accuracy. All frames and cold lazy encoder startup remain charged.
    for offset in [5,9]:
        online('gpu4',18,f'causal_fixed20_offset{offset}_total80_v18_v1','fixed',20,80,20,offset)
    for seed in [20261015,20261016]:
        online('gpu4',18,f'causal_random_seed{seed}_total80_v18_v1','random',10,80,20,scheduler_seed=seed)
    for repetition in [2,3]:
        online('gpu4',18,f'causal_joint_total80_v18_repeat{repetition}_v1','joint',10,80,20)
    # Reconfirm exact full OFF and zero behavior after all implementation hooks.
    for tag,enabled in [('OFF_full_strict_after_all_hooks_v1',False),('ZERO_full_strict_after_all_hooks_v1',True)]:
        command=['geo_tools/run_baseline.py','--video-ids',*map(str,strict),'--tag',tag,'--stream-read']
        if enabled:command+=['--geo-enabled','--feature-weight','0','--geometry-weight','0']
        stage('gpu0',tag,command)
        paths={v:RUNTIME/'baseline'/tag/f'video_{v:02d}.json' for v in strict}
        suite('gpu0',tag+'_suite',paths,[tag])
    raw=[2,3,14,15,16,22,23];tag='B0_raw_diagnostic_remaining7_full_v1'
    stage('gpu0',tag,['geo_tools/run_baseline.py','--video-ids',*map(str,raw),'--tag',tag,'--stream-read'])
    for video in raw:
        stage('gpu0',f'eval_{tag}_v{video}',['geo_tools/evaluate_predictions.py','--prediction',str(RUNTIME/'baseline'/tag/f'video_{video:02d}.json'),
              '--video-id',str(video),'--allow-duplicate-gt'],True,[tag])
    # Additional independent feature controls and budget sensitivity for the
    # real retrieval producer, followed by full ten-stream VGGT comparison.
    retrieval=[]
    for backbone in ['gmt','dinov2']:
        retrieval.append((f'retrieval_{backbone}_scoring_diagnostic_v18_full_v1',[18],backbone,10,True))
    for interval in [5,20]:
        retrieval.append((f'retrieval_vggt_k{interval}_v18_full_v1',[18],'vggt',interval,False))
    retrieval.append(('retrieval_vggt_k10_10streams_full_v1',[18,1,10,11,19],'vggt',10,False))
    for tag,videos,backbone,interval,diagnostic in retrieval:
        command=['geo_tools/run_live_retrieval.py','--video-ids',*map(str,videos),'--tag',tag,'--backbone',backbone,'--interval',str(interval)]
        if diagnostic:command+=['--diagnose-scoring']
        stage('gpu4',tag,command)
        stage('gpu4','eval_'+tag,['geo_tools/evaluate_live_retrieval.py','--run',str(RUNTIME/'retrieval'/tag)],True,[tag])
        if len(videos)>1:
            stage('gpu4','eval_scene18_'+tag,['geo_tools/evaluate_live_retrieval.py','--run',str(RUNTIME/'retrieval'/tag),'--query-scene','18'],True,[tag])
    save_json(ROOT/'manifests/remaining_experiment_matrix_v1.json',dict(id='remaining_matrix_v1',project_gpu_ids=['0','4','5','6'],
              head_seed=20261014,scope='original attachment Phase0-5 remaining experiments; strict5 and original7 invalid-GT baseline diagnostics kept distinct',
              used_test_for_selection=False,realized_budget_not_assumed_from_cap=True,queues=queues,
              scientific_completion_not_implied_by_queue_exit=True))
    print(json.dumps({queue:len(stages) for queue,stages in queues.items()}))


if __name__=='__main__':main()
