"""Actual image/state/RNG future parity after an injected guarded commit fault.

The veto and fault are injected engineering probes, not policy efficacy or GT
evidence. Both arms otherwise use exact raw GMT with passive source metadata.
"""
import argparse
import json
import random
import time
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from common import load_gmt,train_data,video_records,map_records,seed_all,save_json,sha256,CHECKPOINT
from gtr.geometry.online_runner import CausalGMT
from gtr.modeling.meta_arch import gtr_rcnn as native
from gtr.modeling.identity_revision.state_transaction import native_state,fingerprint
from verify_cir_off_parity import rng_digest
from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--groups',type=int,default=40);a=p.parse_args()
    if a.groups<10:raise ValueError('Need actual future after frame8')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);cfg,model=load_gmt();model.min_track_len=0
    rows=video_records(train_data(),4,max_frames=a.groups)
    original_propose=model.propose_identity_revision;original_commit=model.commit_native_identity_state
    original_solver=native.linear_sum_assignment;arms={};started=time.monotonic()
    for arm in ('RAW_PASSIVE','GUARD_FAULT_RECOVERY'):
        seed_all(20261014);controller=model.cir_controller
        controller.enabled=True;controller.mode='guard';controller.audit_only=True
        tracker=CausalGMT(model);state=[];rngs=[];predictions=[];costs=[];probe=dict(veto=False,fault=False)
        model.propose_identity_revision=lambda scores,*unused:scores
        current_costs=[]
        def solve(cost,*args,**kwargs):
            current_costs.append(fingerprint(cost));return original_solver(cost,*args,**kwargs)
        native.linear_sum_assignment=solve
        real_precommit=controller.precommit
        def precommit(scores,accepted,*args,**kwargs):
            if controller.timestamp==8 and not probe['veto'] and kwargs.get('count',True):
                selected=torch.nonzero(accepted>=0,as_tuple=True)[0]
                if len(selected):
                    row=int(selected[0]);candidates=args[2]
                    col=int(torch.nonzero(candidates==accepted[row],as_tuple=True)[0][0])
                    result=scores.clone();result[row,col]=-1e6;probe['veto']=True;controller.veto_edges+=1
                    return result
            return scores
        def commit(*args):
            result=original_commit(*args)
            if probe['veto'] and not probe['fault']:
                probe['fault']=True
                random.random();np.random.random();torch.rand(3);torch.rand(3,device='cuda')
                native.poss_ids.poss_ids.add(999999);native.old_ids.old_ids.add(999998)
                native.old_reids.old_reids.append('injected invalid Bank state')
                raise RuntimeError('engineering fault after genuine native Gallery/count commit')
            return result
        if arm=='GUARD_FAULT_RECOVERY':
            controller.precommit=precommit;model.commit_native_identity_state=commit
        try:
            for frame in sorted({r['frame_id'] for r in rows}):
                current_costs.clear();group=[r for r in rows if r['frame_id']==frame]
                if arm=='GUARD_FAULT_RECOVERY' and frame==8:controller.audit_only=False
                output=tracker.step(map_records(cfg,group))
                for row,item in zip(group,output):
                    predictions.extend(custom_instances_to_coco_json(item['instances'].to('cpu'),row['image_id']))
                snapshot=native_state(tracker,native);snapshot.pop('cir')
                for name in ('cir_window','cir_older'):snapshot['tracker'].pop(name,None)
                state.append(fingerprint(snapshot));rngs.append(rng_digest());costs.append(list(current_costs))
            arms[arm]=dict(predictions=predictions,states=state,rng=rngs,costs=costs,
                fault_injected=probe['fault'],veto_injected=probe['veto'],controller_failed=controller.failed,
                failure_events=controller.failure_events)
            save_json(a.out/(arm+'.json'),arms[arm])
        finally:
            native.linear_sum_assignment=original_solver;model.commit_native_identity_state=original_commit
            model.propose_identity_revision=original_propose
            # Bound method must not leak into the next controller snapshot.
            if 'precommit' in vars(controller):del vars(controller)['precommit']
    baseline=arms['RAW_PASSIVE'];fault=arms['GUARD_FAULT_RECOVERY']
    checks={key:baseline[key]==fault[key] for key in ('predictions','states','rng')}
    checks['all_future_native_costs_equal']=baseline['costs'][8:]==fault['costs'][8:]
    checks['actual_fault_and_veto']=fault['fault_injected'] and fault['veto_injected'] and fault['controller_failed']
    report=dict(status='PASS_REAL_GUARD_COMMIT_ROLLBACK_FUTURE' if all(checks.values()) else 'FAIL_NO_ADMISSION',
        checks=checks,scene=4,groups=a.groups,checkpoint_sha256=sha256(CHECKPOINT),
        script_sha256=sha256(__file__),wall_seconds=time.monotonic()-started,GT_in_inference=False,
        scope='real frozen perception/GTA, Gallery/counts, module Bank, ledger, RNG; full future after one discarded commit',
        not_policy_benefit=True,method_with_fault_invalid=True)
    save_json(a.out/'RESULT.json',report);print(json.dumps(report),flush=True)
    if not all(checks.values()):raise RuntimeError('Guard actual future rollback failed')


if __name__=='__main__':main()
