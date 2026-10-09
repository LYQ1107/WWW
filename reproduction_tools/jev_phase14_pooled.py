"""Freeze all six-camera pooled TrackEval, then perform exact native event taxonomy."""
import argparse
import collections
from jev_phase14_common import *
from run_jev_phase10_closed_loop import metrics
from jev_phase14_forensics import audit_case

METRICS=['HOTA','AssA','IDF1','IDSW','MOTA','Frag']

def pooled(variant,seed,phase):
    protect();source=binding();name=f'{variant}_seed{seed}';root=OUT/f'{phase}_online_v2'/name;out=OUT/f'{phase}_pooled_v2'/name;out.mkdir(parents=True,exist_ok=True)
    path=out/'RESULT.json'
    if path.exists() and (out/'TAXONOMY.json').exists():return read(path)
    records=[read(root/f'video{v:02d}/RESULT.json') for v in DEV];assert all(r['status']=='COMPLETE' for r in records)
    values={};references=[]
    for field,target in [('raw_predictions','RAW_PREDICTIONS.json'),('canonical_predictions','CANONICAL_PREDICTIONS.json')]:
        rows=[]
        for r in records:
            p=r[field];assert sha(p['path'])==p['SHA256'];rows.extend(read(p['path']))
        save(out/target,rows);values[field],evaluation=metrics(out/target,DEV,out/field)
        references.append(dict(kind=field,evaluator=dict(path=str(evaluation/'metrics.json'),SHA256=sha(evaluation/'metrics.json'))))
    result=dict(status='COMPLETE',binding=source,variant=variant,seed=seed,phase=phase,all_six_cameras_pooled=True,per_video_mean_used_as_pooled=False,
        strict_pooled_TrackEval=values['raw_predictions'],canonical_pooled_TrackEval=values['canonical_predictions'],evaluator_references=references,
        actual_video_results=[dict(path=str(root/f'video{v:02d}/RESULT.json'),SHA256=sha(root/f'video{v:02d}/RESULT.json')) for v in DEV],
        developmental_perception_exposure='Stage1 exposed all24TRAIN; association controllers TRAIN12/13/14/16 only')
    save(path,result)
    ann=read('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json');counts=collections.Counter();taxonomies=[]
    for video in DEV:
        t=audit_case(root/f'video{video:02d}',ann,video,artifact_root=OUT/'new_online_forensics_v2'/phase);counts.update(t['counts'])
        taxonomies.append(dict(video=video,counts=t['counts'],exact_CLEAR_switch_reconstruction=t['exact_CLEAR_switch_reconstruction'],event_stream=t['event_stream'],logit_margins=t['logit_margins'],identity_error_intervals=t['identity_error_intervals']))
    save(out/'TAXONOMY.json',dict(status='COMPLETE',binding=source,counts=dict(counts),videos=taxonomies,metrics=result['strict_pooled_TrackEval'],UNKNOWN_not_negative=True))
    print('PHASE14_POOLED_NATIVE_COMPLETE',phase,variant,seed,result['strict_pooled_TrackEval'],flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',default='formal',choices=['formal','onpolicy','offpolicy','oldloss_onpolicy']);a=p.parse_args();pooled(a.variant,a.seed,a.phase)
