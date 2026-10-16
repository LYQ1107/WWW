"""Complete fixed TRAIN01 COW replay against sealed full-snapshot actors.

No GT, metrics, thresholds or scene selection. Engineering qualification only.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import torch
from common import ROOT, load_gmt, train_data, video_records, map_records, seed_all, save_json, sha256, CHECKPOINT
from gtr.geometry.online_runner import CausalGMT
from gtr.modeling.identity_revision.controller import IdentityRevisionController
from gtr.evaluation.mot_evaluation import custom_instances_to_coco_json
from verify_cir_off_parity import SOURCE_AT_IMPORT


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    references={}
    for mode in ('guard','margin'):
        path=a.reference_root/mode/'scene1/video1/manifest.json';r=json.loads(path.read_text())
        if r['status']!='COMPLETE' or not r['full_video'] or r['video_id']!=1 or r['method']!=mode:
            raise ValueError('Fixed complete TRAIN01 reference required')
        if r['checkpoint_sha256']!=sha256(CHECKPOINT) or r['source_commit']!='6f3659cce7472c0fd165967d101ad6a4e9d50bb1':
            raise ValueError('Expected frozen P2 reference source and GMT')
        for spec in r['predictions'].values():
            if sha256(spec['path'])!=spec['sha256']:raise ValueError('Reference output drift')
        references[mode]=(path,r)
    a.out.mkdir(parents=True);cfg,model=load_gmt();stock_min=model.min_track_len;model.min_track_len=0
    model.eval().requires_grad_(False);rows=video_records(train_data(),1)
    expected=sorted(r['image_id'] for r in rows);groups=sorted({r['frame_id'] for r in rows})
    report=dict(status='RUNNING',scene=1,GT_in_inference=False,metrics_read=False,results={},source_sha256_at_import=SOURCE_AT_IMPORT)
    for mode,(path,reference) in references.items():
        seed_all(reference['seed']);cc=cfg.MODEL.CIR.clone();cc.defrost();cc.ENABLED=True;cc.MODE=mode;cc.COMMIT_SNAPSHOT='cow_engineering'
        model.cir_controller=IdentityRevisionController(cc);tracker=CausalGMT(model)
        predictions=[];processed=[];started=time.monotonic()
        for index,frame in enumerate(groups):
            current=[r for r in rows if r['frame_id']==frame]
            with torch.inference_mode():outputs=tracker.step(map_records(cfg,current))
            for row,output in zip(current,outputs):
                processed.append(row['image_id']);predictions.extend(custom_instances_to_coco_json(output['instances'].to('cpu'),row['image_id']))
            if index%50==0:
                report.update(current=mode,completed_groups=index+1,total_groups=len(groups));save_json(a.out/'RESULT.json',report)
                print(json.dumps(dict(mode=mode,groups=index+1,total=len(groups))),flush=True)
        torch.cuda.synchronize();seconds=time.monotonic()-started
        count=Counter(r['track_id'] for r in predictions);stock=[r for r in predictions if count[r['track_id']]>=stock_min]
        controller=model.cir_controller
        checks=dict(images=sorted(processed)==expected==sorted(reference['processed_image_ids']),
            raw_predictions=predictions==json.loads(Path(reference['predictions']['raw']['path']).read_text()),
            stock_predictions=stock==json.loads(Path(reference['predictions']['stock']['path']).read_text()),
            no_commit_failure=not controller.failed,no_missing_evidence=not controller.fallbacks,
            actual_veto_exercised=controller.veto_edges>0,
            veto_count=controller.veto_edges==reference['CIR_stats']['guard_veto_edges'],
            native_solves=controller.guard_joint_solves==reference['CIR_stats']['guard_joint_solves'],
            native_validation_aborts=controller.guard_aborted==reference['CIR_stats']['guard_aborted'])
        save_json(a.out/(mode+'_raw.json'),predictions);save_json(a.out/(mode+'_stock.json'),stock)
        report['results'][mode]=dict(checks=checks,images=len(processed),groups=len(groups),
            reference_manifest=dict(path=str(path),sha256=sha256(path)),
            reference_full_image_seconds=reference['full_image_seconds'],COW_replay_seconds=seconds,
            timing_caveat='Different entrypoints and current hardware load; engineering wall time only, not formal comparative FPS',
            veto_edges=controller.veto_edges,constrained_solves=controller.guard_joint_solves)
        save_json(a.out/'RESULT.json',report)
        if not all(checks.values()):raise RuntimeError('Complete native COW replay mismatch: '+mode)
    report.update(status='PASS_COMPLETE_FIXED_SCENE_COW_REFERENCE_OUTPUTS',full_video=True,
        formal_admission=False,scope='Fixed TRAIN01 guard and margin; real mutated-state CausalGMT; all physical images and exact outputs vs sealed original full-snapshot actors. No GT or metric selection.')
    save_json(a.out/'RESULT.json',report)


if __name__=='__main__':main()
