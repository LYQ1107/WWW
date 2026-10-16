"""Bounded actual-image paired intervention at every sealed FIT SUSPECT source.

One causal original prefix per scene; restore identical state/RNG before each
C1 versus single-target Guard future. No GT, teacher, cache or ideal candidates.
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
import subprocess
import time
import torch
from common import ROOT,load_gmt,train_data,video_records,map_records,seed_all,save_json,sha256,CHECKPOINT
from gtr.geometry.online_runner import CausalGMT
from gtr.modeling.meta_arch import gtr_rcnn as native
from gtr.modeling.identity_revision.state_transaction import StateTransaction,native_state,fingerprint
from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
from verify_cir_off_parity import rng_digest

EVENTS_SHA='106f5e7efce7b0ed08bb1f327d8aa30696ef65dbb5b29fadbdaeec88a44edc74'


def load_targets(path):
    if sha256(path)!=EVENTS_SHA:raise ValueError('Sealed FIT policy event SHA mismatch')
    with gzip.open(path,'rt') as stream:
        targets=[json.loads(line) for line in stream if 'SUSPECT' in line]
    targets=[r for r in targets if r['evidence_state']=='SUSPECT']
    if len(targets)!=26 or {r['key'][0] for r in targets}!={12,20,21,24}:
        raise ValueError('All26 predeclared sources required; no outcome selection')
    if len({tuple(r['key']) for r in targets})!=26:raise ValueError('Repeated physical target')
    return targets


def public_rows(group,output):
    rows=[]
    for record,item in zip(group,output):
        rows.extend(custom_instances_to_coco_json(item['instances'].to('cpu'),record['image_id']))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);a=p.parse_args()
    protocol=json.loads(a.protocol.read_text())
    if protocol['horizon_camera_groups']!=64 or protocol['events_sha256']!=EVENTS_SHA:
        raise ValueError('Frozen paired intervention protocol required')
    proof=protocol['native_guard_prefix']
    if sha256(proof['path'])!=proof['sha256']:raise ValueError('Guard prefix proof changed')
    admission=json.loads(Path(proof['path']).read_text())
    if admission['status']!='PASS_REAL_PREFIX_GUARD_OFFICIAL_CAUSAL':raise ValueError('Native shared-core proof required')
    for name,digest in admission['source_sha256_at_import'].items():
        if name.startswith('gtr/') and sha256(ROOT/name)!=digest:raise ValueError('Qualified native source drift: '+name)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('Clean frozen paired source required')
    targets=load_targets(a.events)
    if a.scene not in (12,20,21,24):raise ValueError('Only predeclared FIT target scenes')
    targets=sorted([r for r in targets if r['key'][0]==a.scene],key=lambda r:(r['frame'],r['key']))
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    cfg,model=load_gmt();model.min_track_len=0;controller=model.cir_controller
    controller.enabled=True;controller.mode='guard';controller.audit_only=True
    seed_all(20261014);tracker=CausalGMT(model)
    rows=video_records(train_data(),a.scene);groups=defaultdict(list)
    for row in rows:groups[row['frame_id']].append(row)
    frames=sorted(groups);by_frame=defaultdict(list)
    for target in targets:by_frame[target['frame']].append(target)
    prefix=[];completed=[];started=time.monotonic()
    manifest=dict(status='RUNNING',scene=a.scene,GT_in_inference=False,
        perception_cache_used=False,checkpoint_sha256=sha256(CHECKPOINT),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
        protocol=dict(path=str(a.protocol),sha256=sha256(a.protocol)),targets=len(targets),completed=[])
    save_json(a.out/'manifest.json',manifest)
    try:
        for frame in frames:
            if frame>max(by_frame):break
            for target in by_frame.get(frame,[]):
                key=tuple(target['key']);label='_'.join(map(str,key));case=a.out/label;case.mkdir()
                # Snapshot complete original native state immediately before
                # the synchronized group, including RNG and passive ledger.
                tx=StateTransaction(tracker,native);initial_rng=rng_digest();branches={}
                horizon=[f for f in frames if frame<=f<=frame+64]
                baseline_compute=model.compute_native_track_scores
                for arm in ('C1_KEEP','C1_SINGLE_TARGET_GUARD'):
                    tx.restore()
                    if fingerprint(native_state(tracker,native))!=tx.before_digest or rng_digest()!=initial_rng:
                        raise RuntimeError('Branches do not share exact native initial state/RNG')
                    controller.audit_only=False;controller.guard_target_sources={key} if arm.endswith('GUARD') else set()
                    inputs=[];predictions=[];t=time.monotonic()
                    def compute(*args,**kwargs):
                        result=baseline_compute(*args,**kwargs)
                        instances=args[1];k=args[3];query=instances[k]
                        if len(query) and int(query.cir_source[0,3])==key[1]:
                            if key[2]>=len(query):raise ValueError('Physical target row disappeared')
                            refs=[inst for i,inst in enumerate(instances) if i!=k]
                            payload=dict(query=query.reid_features,source=query.cir_source,
                                history=torch.cat([x.reid_features for x in refs]),
                                history_ids=torch.cat([x.track_ids for x in refs]),
                                history_source=torch.cat([x.cir_source for x in refs]),
                                candidates=result[1],native_scores=result[3],counts=tracker.counts,
                                Bank=native.old_reids.old_reids,poss_ids=native.poss_ids.poss_ids)
                            inputs.append(dict(input_sha256=fingerprint(payload),rng_sha256=rng_digest(),
                                candidate_ids=result[1].cpu().tolist(),native_scores=result[3].cpu().tolist()))
                        return result
                    model.compute_native_track_scores=compute
                    old_veto=controller.veto_edges;old_solves=controller.guard_joint_solves
                    try:
                        for future in horizon:
                            # Only the target group may intervene. Later states
                            # are genuinely mutated and GTA recomputes from them.
                            if future!=frame:controller.guard_target_sources=set()
                            output=tracker.step(map_records(cfg,groups[future]))
                            predictions.extend(public_rows(groups[future],output))
                            if controller.failed:raise RuntimeError('Guard native invariant failed; no efficacy admission')
                    finally:model.compute_native_track_scores=baseline_compute
                    torch.cuda.synchronize()
                    save_json(case/(arm+'_predictions.json'),predictions)
                    branches[arm]=dict(inputs=inputs,veto_edges=controller.veto_edges-old_veto,
                        guard_joint_solves=controller.guard_joint_solves-old_solves,
                        wall_seconds=time.monotonic()-t,predictions=dict(path=str(case/(arm+'_predictions.json')),
                        sha256=sha256(case/(arm+'_predictions.json'))),final_rng=rng_digest())
                same_input=branches['C1_KEEP']['inputs']==branches['C1_SINGLE_TARGET_GUARD']['inputs']
                if not same_input or len(branches['C1_KEEP']['inputs'])!=1:
                    raise RuntimeError('Actual target GTA/candidate/evidence/RNG input differs across paired arms')
                report=dict(status='COMPLETE_PAIRED_NATIVE_FUTURE',target=target,
                    prefix_predictions_before_target=len(prefix),initial_state_sha256=tx.before_digest,
                    initial_rng_sha256=initial_rng,identical_target_input=True,
                    first_frame=frame,last_frame=horizon[-1],requested_future_groups=64,
                    actual_future_groups=len(horizon)-1,end_censored=len(horizon)<65,
                    branches=branches,GT_in_inference=False,not_full_video_metrics=True)
                save_json(case/'RESULT.json',report);completed.append(dict(key=key,result=str(case/'RESULT.json'),sha256=sha256(case/'RESULT.json')))
                tx.restore();del tx
                manifest.update(completed=completed,elapsed_seconds=time.monotonic()-started)
                save_json(a.out/'manifest.json',manifest)
                print(json.dumps(dict(scene=a.scene,target=key,completed=len(completed),total=len(targets),
                    veto_edges=branches['C1_SINGLE_TARGET_GUARD']['veto_edges'],elapsed=manifest['elapsed_seconds'])),flush=True)
            output=tracker.step(map_records(cfg,groups[frame]));prefix.extend(public_rows(groups[frame],output))
            if frame%25==0:
                manifest.update(prefix_frame=frame,completed=completed,elapsed_seconds=time.monotonic()-started)
                save_json(a.out/'manifest.json',manifest)
                print(json.dumps(dict(scene=a.scene,prefix_frame=frame,cases=len(completed),elapsed=manifest['elapsed_seconds'])),flush=True)
        save_json(a.out/'original_prefix_predictions.json',prefix)
        manifest.update(status='COMPLETE_ALL_PREDECLARED_SCENE_TARGETS',
            prefix=dict(path=str(a.out/'original_prefix_predictions.json'),sha256=sha256(a.out/'original_prefix_predictions.json')),
            completed=completed,elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        manifest.update(status='FAILED_NO_ADMISSION',error=repr(error));raise
    finally:save_json(a.out/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],scene=a.scene,cases=len(completed))),flush=True)


if __name__=='__main__':main()
