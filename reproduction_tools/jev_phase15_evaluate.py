"""Strict unfiltered complete native videos, with explicit live/cache provenance."""
import argparse
import collections
import gzip
import time
import torch
from jev_phase15_common import *
from jev_phase13_runtime import build_tracker,cache_inputs,run
from jev_phase14_native_risk import NativeRisk
from gtr.modeling.jev_phase15.native_commit_adapter import attach
from gtr.modeling.jev_phase15.model import PersistentIdentityPolicy
from gtr.modeling.jev_native_state import fingerprint
from run_jev_phase10_closed_loop import raw_predictions,metrics


def owner_propagation(trace):
    """Actual committed errors persist after mixtures; gaps remain censored."""
    owners={};episodes=[];active={};counts=collections.Counter()
    for item in trace:
        frame,view=item['key'];gt=item['GT_OFFLINE_ONLY'];identity=item['identity']
        if gt is None:counts['unassessed_current_GT']+=1;continue
        if identity not in owners:owners[identity]=gt
        wrong=owners[identity]!=gt;counts['assessed_observations']+=1;counts['wrong_fixed_owner_observations']+=wrong
        key=gt,view;old=active.get(key)
        if wrong:
            if old is not None and frame>old['end']+1:
                episodes.append(dict(old,right_censored=True,end_reason='observation gap'));old=None
            if old is None:old=dict(GT_OFFLINE_ONLY=gt,view=view,start=frame,end=frame,observed_frames=1)
            else:old.update(end=frame,observed_frames=old['observed_frames']+1)
            active[key]=old
        elif old is not None:
            episodes.append(dict(old,right_censored=frame!=old['end']+1,
                end_reason='actual certified next-frame correct-owner commit' if frame==old['end']+1 else 'gap before correction'))
            del active[key]
    episodes.extend(dict(item,right_censored=True,end_reason='video end') for item in active.values())
    return dict(counts=dict(counts),episodes=episodes,
        semantics='immutable first-observed GT owner for each native ID; continued errors counted after gallery mixtures; camera/frame observation gaps censored',
        scope='offline observed identity-confusion durations on actual committed IDs; IoU>=0.5 matching coverage disclosed; not interchangeable with CLEAR IDSW or global IDF1')


def load_policy(arm,seed,version,phase):
    path=OUT/f'training_full_payload_v{version}'/arm/f'seed{seed}'/phase/'RESULT.json'
    result=read(path);assert result['status']=='COMPLETE'
    checkpoint=result['checkpoint'];assert sha(checkpoint['path'])==checkpoint['SHA256']
    policy=PersistentIdentityPolicy(arm).cuda().eval()
    weights=torch.load(checkpoint['path'],map_location='cpu');policy.load_state_dict(weights['model'],strict=True)
    policy.posterior_feedback=weights.get('posterior_feedback','native')
    return policy,checkpoint,ref(path)


def main(video,arm='F_full',seed=20261009,version=1,phase='pilot',live=False):
    protect();storage_guard();assert video in DEV
    torch.set_num_threads(1);torch.manual_seed(20261009)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    policy,checkpoint,training=load_policy(arm,seed,version,phase)
    values,frames,reader=cache_inputs(video)
    model=build_tracker(video,policy,react_learned=False,live=live);executor=attach(model)
    out=OUT/f'{phase}_online_v{version}'/f'{arm}_seed{seed}'/('live' if live else 'frozen_perception')/f'video{video:02d}'
    out.mkdir(parents=True,exist_ok=True);assert not (out/'RESULT.json').exists()
    source=binding(seed=seed,checkpoints=[checkpoint],dataset=ref(ANNOTATIONS),
        evaluator='post-freeze TrackEval raw predictions plus actual-commit offline risk',
        scope='complete previously reused DEVELOPMENT; no sealed evaluation or independent test claim')
    source['perception_input_provenance']=perception_provenance(video)
    source['perception_execution']='current images through actual Stage1 detector/VFCE' if live else 'frozen Stage1 cache'
    risk=NativeRisk(video,reader);start=time.monotonic();last_progress=[0.]
    journal=gzip.open(out/'QUESTIONS.jsonl.gz','wt');commits=gzip.open(out/'COMMITS.jsonl.gz','wt')

    def before(**d):
        risk.before(**d);c=d['context']
        journal.write(json.dumps(dict(key=[video,c['frame'],c['view']],task=d['task'],refs=d['refs'],
            rows=c.get('rows',list(range(len(d['logits'])))),logits=d['logits'].cpu().tolist(),legal=d['batch']['legal'][0].cpu().tolist()))+'\n')

    def after(**d):
        risk.after(**d);ids=d['instances'][-1].track_ids.cpu().tolist()
        assert len(ids)==len(set(ids));commits.write(json.dumps(dict(key=[video,d['frame'],d['view']],ids=ids,events=d['events'],id_count=d['id_count']))+'\n')
        if time.monotonic()-last_progress[0]>=15:
            save(out/'PROGRESS.json',dict(status='RUNNING',key=[video,d['frame'],d['view']],frames=frames,
                counts=dict(risk.counts),seconds=time.monotonic()-start));journal.flush();commits.flush();last_progress[0]=time.monotonic()
    executor.observer=before;executor.commit_observer=after
    try:
        with torch.no_grad():raw,_=run(model,values,frames)
    finally:journal.close();commits.close()
    assert risk.counts['payloads']==2*frames-1
    predictions=raw_predictions(raw,risk.labels.images);save(out/'RAW_PREDICTIONS.json',predictions)
    identity=owner_propagation(risk.trace);save(out/'ERROR_PROPAGATION.json',identity)
    strict,evaluation=metrics(out/'RAW_PREDICTIONS.json',[video],out/'strict_eval')
    save(out/'RESULT.json',dict(status='COMPLETE',binding=source,video=video,frames=frames,arm=arm,seed=seed,version=version,phase=phase,
        live_images=live,trained=training,checkpoint=checkpoint,raw_predictions=ref(out/'RAW_PREDICTIONS.json'),
        strict_online_metrics=strict,evaluator_result=ref(evaluation/'metrics.json'),native_risk=risk.summary(),
        error_propagation=ref(out/'ERROR_PROPAGATION.json'),commits=ref(out/'COMMITS.jsonl.gz'),questions=ref(out/'QUESTIONS.jsonl.gz'),
        final_identity_and_commitment_memory_SHA256=fingerprint(executor.memory.state_dict()),
        native_Gallery_Bank_updates=True,short_track_filter_or_GT_renumbering=False,actual_mutated_state_online=True,
        full_FPS=None,latency_scope='instrumented validation; no deployment FPS claim',seconds=time.monotonic()-start))
    save(out/'PROGRESS.json',dict(status='COMPLETE',frames=frames,strict_metrics=strict))
    print('PHASE15_FULL_NATIVE_VIDEO_COMPLETE',arm,version,phase,video,strict,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--arm',default='F_full');p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--version',type=int,default=1);p.add_argument('--phase',default='pilot');p.add_argument('--live',action='store_true');a=p.parse_args()
    main(a.video,a.arm,a.seed,a.version,a.phase,a.live)
