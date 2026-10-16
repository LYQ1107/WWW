"""Full real failure-scene Guard parity with bounded versus wide metadata.

Same actual native images, Guard policy and RNG. Ledger capacity must never
change perception, costs, native identities, Gallery/counts/Bank or the future.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from common import load_gmt,train_data,video_records,map_records,seed_all,save_json,sha256,CHECKPOINT
from gtr.geometry.online_runner import CausalGMT
from gtr.modeling.meta_arch import gtr_rcnn as native
from gtr.modeling.identity_revision.state_transaction import native_state,fingerprint
from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
from verify_cir_off_parity import SOURCE_AT_IMPORT,decisions,rng_digest
from cir_score_trace import C1ScoreTrace


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);cfg,model=load_gmt();model.min_track_len=0
    rows=video_records(train_data(),1);frames=sorted({r['frame_id'] for r in rows});results={};begun=time.monotonic()
    for limit in (32,1024):
        seed_all(20261014);controller=model.cir_controller
        controller.enabled=True;controller.mode='guard';controller.audit_only=False;controller.max_fragments=limit
        tracker=CausalGMT(model);path=a.out/str(limit);path.mkdir();trace=C1ScoreTrace(model,path);trace.install()
        predictions=[];states=[];rng=[]
        try:
            for frame in frames:
                group=[r for r in rows if r['frame_id']==frame]
                output=tracker.step(map_records(cfg,group))
                if controller.failed:raise RuntimeError('Guard invariant failed in full ledger regression')
                controller.ledger.validate_invariants()
                for record,item in zip(group,output):
                    predictions.extend(custom_instances_to_coco_json(item['instances'].to('cpu'),record['image_id']))
                state=native_state(tracker,native);state.pop('cir')
                states.append(fingerprint(state));rng.append(rng_digest())
                if frame%50==0:
                    print(json.dumps(dict(limit=limit,frame=frame,total=len(frames),evictions=controller.ledger.evictions)),flush=True)
        finally:trace.close()
        results[limit]=dict(predictions=predictions,states=states,rng=rng,decisions=decisions(path),evictions=controller.ledger.evictions)
        save_json(path/'predictions.json',predictions);save_json(path/'states.json',states)
    checks={key:results[32][key]==results[1024][key] for key in ('predictions','states','rng','decisions')}
    checks['actual_bounded_eviction']=results[32]['evictions']>0
    result=dict(status='PASS_FULL_NATIVE_LEDGER_RETENTION' if all(checks.values()) else 'FAIL_NO_ADMISSION',
        comparisons=checks,bounded_evictions=results[32]['evictions'],wide_evictions=results[1024]['evictions'],
        scene=1,full_video=True,groups=len(frames),actual_images_per_arm=len(rows),
        source_sha256_at_import=SOURCE_AT_IMPORT,this_script_sha256=sha256(__file__),checkpoint_sha256=sha256(CHECKPOINT),
        GT_in_inference=False,scientific_success=False,wall_seconds=time.monotonic()-begun,
        state_scope='all tracker Gallery/counts/Bank and RNG; auxiliary ledger intentionally differs in retained ended metadata only',
        policy='same fixed Guard/C1 and actual candidate/native solvers, no threshold change')
    save_json(a.out/'RESULT.json',result);print(json.dumps(result),flush=True)
    if not all(checks.values()):raise RuntimeError('Ledger capacity changed actual native future')


if __name__=='__main__':main()
