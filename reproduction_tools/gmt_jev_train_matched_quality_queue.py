"""Six finite full TRAIN controls and pooled metrics; never select a sequence."""
import gzip
from gmt_jev_common import *
from gmt_jev_evaluate import evaluate
from gmt_jev_speed_queue import publish_step


def main():
    oracle_job=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs']['P1_bounded_oracle_all_TRAIN_GPU7']
    assert oracle_job['status']=='COMPLETE'
    queue=OUT/'train_matched_quality/QUEUE.json'
    for video in [12,13,14]:
        for kind in ['original','rule']:
            name=f'P4_TRAIN{video}_{kind}_matched_full_GPU7'
            jobs=read(REPORTS/'EXPERIMENT_REGISTRY.json')['jobs']
            if name in jobs:
                assert jobs[name]['status']=='COMPLETE','Never duplicate a nonterminal job'
                continue
            assert not any(d['status']=='RUNNING' and d.get('GPU')==7 for d in jobs.values())
            save(queue,dict(status='RUNNING_MATCHED_TRAIN_QUALITY',PID=os.getpid(),
                            video=video,kind=kind,physical_GPU=7,job=name))
            subprocess.run([PYTHON,str(ROOT/'reproduction_tools/gmt_jev_job.py'),
                            '--name',name,'--gpu','7','--script','gmt_jev_train_matched_quality.py',
                            '--','--video',str(video),'--kind',kind],cwd=ROOT,check=True)
            publish_step(f'full TRAIN{video} matched {kind} trajectory quality')
    methods={}
    save(queue,dict(status='RUNNING_POOLED_TRAIN_METRICS',PID=os.getpid()))
    for kind in ['original','rule']:
        cases=[read(REPORTS/f'TRAIN_MATCHED_{kind}_VIDEO{v:02d}.json') for v in [12,13,14]]
        pooled={}
        for convention in ['RAW','STOCK']:
            rows=[]
            for case in cases:
                reference=case['inference']['predictions'][convention]
                assert sha(reference['path'])==reference['SHA256']
                with gzip.open(reference['path'],'rt') as f:
                    rows.extend(json.load(f))
            pooled[convention]=evaluate(rows,OUT/'train_matched_quality/pooled'/kind/convention,videos=[12,13,14])
        methods[kind]=dict(cases=[ref(REPORTS/f'TRAIN_MATCHED_{kind}_VIDEO{v:02d}.json') for v in [12,13,14]],
                           pooled=pooled,
                           primary_descriptive_duration_camera_frames=sum(c['primary_identity']['identity_inconsistency_duration_camera_frames'] for c in cases))
    before=methods['original']['pooled']['RAW'];after=methods['rule']['pooled']['RAW']
    deltas={key:after['metrics'][key]-value for key,value in before['metrics'].items()}
    deltas['CVIDF1']=after['CVIDF1']-before['CVIDF1']
    save(REPORTS/'FULL_TRAIN_MATCHED_QUALITY_AUDIT.json',dict(
        status='COMPLETE_ALL_REGISTERED_TRAIN_MATCHED_FULL_TRAJECTORY_QUALITY',
        videos=[12,13,14],methods=methods,primary_pooled_deltas=deltas,
        no_gradient_updates=True,training_qualification_only=True,
        independent_validation=False,scientific_success=False,
        accepted_reference_native_counter_results_preserved=ref(REPORTS/'FULL_TRAIN_NATIVE_RULE_AUDIT.json'),
        interpretation='Whole cold trajectories under fixed original GMT frontend/native rules. Pooled official metrics across all three registered TRAIN videos; no favorable-sequence selection, causal N01/N10 or independent-validation claim. Existing TRAIN16 primary failures remain unchanged.',
        source=binding()))
    publish_step('all registered TRAIN pooled matched trajectory quality')
    save(queue,dict(status='COMPLETE_FINITE_MATCHED_TRAIN_QUALITY',PID=os.getpid()))


if __name__=='__main__':
    main()
