"""Real images, original source / CIR OFF / KEEP / branch rollback qualification.

No GT labels are passed to inference. Prefix evidence is not a full-video claim.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import random
import time
from types import MethodType
import numpy as np
import torch
from common import ROOT, load_gmt, map_records, train_data, video_records, seed_all, save_json, sha256, CHECKPOINT
from test_cir_native_commit import original_method
from gtr.modeling.meta_arch import gtr_rcnn as native
from gtr.geometry.online_runner import CausalGMT
from gtr.geometry.decision_trace import DecisionTrace
from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
from gtr.modeling.identity_revision.state_transaction import StateTransaction, native_state, fingerprint

SOURCE_AT_IMPORT={str(p.relative_to(ROOT)):sha256(p) for p in (
    ROOT/'geo_tools/verify_cir_off_parity.py', ROOT/'geo_tools/test_cir_native_commit.py',
    ROOT/'gtr/config.py', ROOT/'gtr/modeling/meta_arch/gtr_rcnn.py',
    ROOT/'gtr/geometry/online_runner.py', ROOT/'gtr/geometry/decision_trace.py',
    *sorted((ROOT/'gtr/modeling/identity_revision').glob('*.py')))}


class FullScoreTrace(DecisionTrace):
    def solve(self,cost,*args,**kwargs):
        result=super().solve(cost,*args,**kwargs)
        if self.stack:
            self.stack[-1]['records'][-1]['full_native_cost_sha256']=fingerprint(cost)
        return result


def decisions(directory):
    result=[]
    for path in (directory/'actual_decisions.jsonl.gz', directory/'after_rollback/actual_decisions.jsonl.gz'):
        if not path.exists():continue
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);row.pop('method_call_index',None);result.append(row)
    return result


def rng_digest():
    return fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state(),
                        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',type=int,default=4)
    p.add_argument('--groups',type=int,default=40)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.scene not in range(1,25):raise ValueError('TRAIN only')
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    started=time.monotonic()
    cfg,model=load_gmt()
    data=train_data();records=video_records(data,a.scene,max_frames=a.groups)
    rows_by_frame={f:[r for r in records if r['frame_id']==f] for f in sorted({r['frame_id'] for r in records})}
    original=original_method();modified=native.GTRRCNN.run_global_tracker_plus
    results={}
    for arm in ('ORIGINAL_SOURCE','CIR_OFF','CIR_KEEP'):
        seed_all(20261014)
        model.run_global_tracker_plus=MethodType(original if arm=='ORIGINAL_SOURCE' else modified,model)
        model.cir_controller.enabled=arm=='CIR_KEEP'
        tracker=CausalGMT(model)
        directory=a.out/arm;directory.mkdir()
        trace=FullScoreTrace(model,directory);trace.install()
        states=[];predictions=[];rngs=[];rollback_checked=False
        t=time.monotonic()
        try:
            for frame,rows in rows_by_frame.items():
                mapped=map_records(cfg,rows)
                # Re-execute a real native/image step after full alias-preserving restore.
                if arm=='CIR_OFF' and frame==min(8,len(rows_by_frame)):
                    tx=StateTransaction(tracker,native)
                    # Observer has external open files; disable it for the discarded branch.
                    trace.close()
                    unobserved_inference=model.inference
                    def branch_inference(*args,**kwargs):
                        result=unobserved_inference(*args,**kwargs)
                        for record,inst in zip(args[0],result):
                            n=len(inst);device=inst.reid_features.device
                            inst.decision_trace_source=torch.stack((
                                torch.full((n,),int(record['image_id']),device=device,dtype=torch.long),
                                torch.arange(n,device=device)),dim=1)
                        return result
                    # The passive provenance field must remain schema-compatible
                    # with the already-observed real history, even on discard.
                    model.inference=branch_inference
                    try:
                        tracker.step(copy.deepcopy(mapped))
                    finally:
                        model.inference=unobserved_inference
                    tx.restore()
                    # Resume a fresh observer file, preserving the primary trace separately.
                    extra=directory/'after_rollback';extra.mkdir()
                    trace=FullScoreTrace(model,extra);trace.install()
                    rollback_checked=True
                output=tracker.step(mapped)
                for row,item in zip(rows,output):
                    predictions.extend(custom_instances_to_coco_json(item['instances'].to('cpu'),row['image_id']))
                state=native_state(tracker,native);state.pop('cir')
                # Runtime provenance tags are passive; include their real values in state.
                states.append(fingerprint(state));rngs.append(rng_digest())
        finally:
            if trace is not None:trace.close()
        torch.cuda.synchronize()
        results[arm]=dict(predictions=predictions,states=states,rngs=rngs,seconds=time.monotonic()-t,
                          rollback_checked=rollback_checked,decisions=decisions(directory))
        save_json(directory/'predictions.json',predictions)
        save_json(directory/'states.json',states)
        print(json.dumps(dict(arm=arm,groups=len(states),seconds=results[arm]['seconds'])),flush=True)
    # Passive observer resets source tags per image. Physical keys are deterministic.
    comparisons={}
    for arm in ('CIR_OFF','CIR_KEEP'):
        comparisons[arm]={key:results[arm][key]==results['ORIGINAL_SOURCE'][key] for key in ('predictions','states','rngs','decisions')}
    report=dict(status='PASS_REAL_PREFIX_OFF_KEEP_ROLLBACK' if all(all(c.values()) for c in comparisons.values()) else 'FAIL_NO_ADMISSION',
        scene=a.scene,groups=len(rows_by_frame),full_video=len(records)==len(video_records(data,a.scene)),
        comparisons=comparisons,checkpoint_sha256=sha256(CHECKPOINT),
        source_sha256_at_import=SOURCE_AT_IMPORT,
        config_sha256=sha256(ROOT/'configs/VISION_test.yaml'),
        image_sha256={str(r['image_id']):sha256(r['file_name']) for r in records},
        wall_seconds=time.monotonic()-started,training=False,GT_in_inference=False,
        state_scope='all real tracker attributes except model; complete Instances/counts/reids, module poss/old/Bank; CIR own call counter excluded for KEEP',
        rollback_scope='one discarded actual image/native step then restored; equality assessed by all remaining outputs/states/RNG against original source',
        official_sliding_parity='pending separately; prefix CausalGMT only',
        arms={k:{'seconds':v['seconds'],'rollback_checked':v['rollback_checked']} for k,v in results.items()})
    save_json(a.out/'RESULT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='image_sha256'}),flush=True)
    if report['status'].startswith('FAIL'):raise RuntimeError('OFF/KEEP/rollback parity failed')


if __name__=='__main__':main()
