"""All3000 old TRAIN inputs/labels parity before any full legacy-protocol fit."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from cv_pil_v3_fit import load_fit
from cv_pil_native_handoff import inference_inputs
from cv_pil_v2_model import tensorize,training_eligible,CandidatePredictor
from pirga_legacy_preprocessing import LegacyTrainingCache
from pirga_fulltrain_dataset import FullTrainDataset,load_supervision
from pirga_grouped_heads import GroupedHeads,grouped_losses
from pirga_indexed_store import sha


def run(preparation_path,out,canonical_pair=False):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();torch.set_num_threads(1);ad=json.loads(preparation_path.read_text());cache=LegacyTrainingCache(ad['input_manifest']['path'],ad['input_manifest']['sha256'])
    a=json.loads(Path(ad['full_admission']['path']).read_text());ds=FullTrainDataset(a['stores']);labels=load_supervision(a['supervision']['path'],a['supervision']['sha256'],ds)
    repo=Path(__file__).resolve().parents[1];old_protocol_path=repo/'results/cv_pil_v3_native_validation/P3_MATCHED_FIT_PREREGISTRATION.json';old=json.loads(old_protocol_path.read_text());events,audit=load_fit(**old['fit_source'])
    tested=0;old_eligible=0;old_cross=0;label_differences=[];training_differences=[];eligible=set(map(tuple,json.loads(Path(ad['eligibility']['path']).read_text())))
    for event in events:
        actor=inference_inputs(event);key=tuple(actor['event_key']);normal=tensorize(actor);fast=cache.batch(key);new=cache.label(key,labels[key]);original=event['OFFLINE_ONLY']
        expected_ids=list(actor['native_candidate_ids']);actual_ids=[ds.candidate_ids(key)[i] for i in cache.columns(key)]
        if expected_ids!=actual_ids:raise ValueError('Actual native Top8 identity columns differ '+str(key))
        for field in ('query','histories','owners','age','cross_view','native_scores'):
            if not torch.equal(getattr(normal,field),getattr(fast,field)):raise ValueError('Old input field differs '+str((key,field)))
        if normal.destination_camera!=fast.destination_camera:raise ValueError('Destination camera differs')
        ok=training_eligible(normal,original);new_ok=key in eligible
        different=(list(original['certified_target_indices'])!=new['certified_target_indices'] or list(original['candidate_known_mask'])!=new['candidate_known_mask'] or original['GT_person']!=new['GT_person'])
        if different or ok!=new_ok:
            detail=dict(key=list(key),old_eligible=ok,new_eligible=new_ok,old=dict(GT_person=original['GT_person'],certified_target_indices=list(original['certified_target_indices']),candidate_known_mask=list(original['candidate_known_mask'])),new=new)
            label_differences.append(detail)
            if ok or new_ok:training_differences.append(detail)
        old_eligible+=ok;old_cross+=bool(original['has_strict_past_cross_view_positive']);tested+=1
        if tested%500==0:print(json.dumps(dict(old_TRAIN_queries_verified=tested,DEV_read=False)),flush=True)
    if tested!=3000 or old_eligible!=2592:raise ValueError('Frozen old TRAIN coverage changed')
    if training_differences and not canonical_pair:
        failure=dict(status='FAILED_OLD_TRAIN_OBJECTIVE_CERTIFICATION_PARITY',old_queries_tested=tested,old_ranking_eligible=old_eligible,batch_fields_bitwise_equal=True,
            all_label_differences=label_differences,training_objective_differences=training_differences,training_updates=0,GPU_wall_hours=0,DEV_read=False,
            preparation_sha256=sha(preparation_path),source_sha256=sha(__file__),wall_seconds=time.monotonic()-start)
        out.write_text(json.dumps(failure,indent=2)+'\n');raise ValueError('Old training masks differ; preserve evidence and do not train')
    if canonical_pair:
        for item in label_differences:
            if item['old']['GT_person']!=item['new']['GT_person'] or item['old']['certified_target_indices']!=item['new']['certified_target_indices'] or item['old_eligible']!=item['new_eligible'] or any(b and not a for a,b in zip(item['old']['candidate_known_mask'],item['new']['candidate_known_mask'])):
                raise ValueError('Paired canonical control cannot repair changed truth/positives/eligibility or upgrade UNKNOWN negatives')
    # Existing model-wise independent-head math already passed real shadow
    # tests. Measure only bounded real TRAIN optimizer throughput here; these
    #22 probe updates per arm are discarded and never used as trained weights.
    keys=sorted(eligible);selected=[]
    for scene in sorted(ds.scenes):
        local=[k for k in keys if k[0]==scene];selected.extend([local[len(local)//2],max(local,key=lambda k:int((cache.calls[cache.locations[k][0]]['positions'][cache.locations[k][1]]>=0).sum()))])
    group=GroupedHeads(seed=20261015);opt=torch.optim.AdamW(group.parameters(),lr=.001,weight_decay=.0001);times=[]
    for key in selected:
        b=cache.batch(key);label=cache.label(key,labels[key]);begin=time.monotonic();score,h=group(b);loss,_,_,_=grouped_losses(score,h,b,label)
        if not torch.isfinite(loss).all():raise ValueError('Nonfinite TRAIN probe loss')
        opt.zero_grad(set_to_none=True);loss.sum().backward();group.clip_per_arm();opt.step();times.append(time.monotonic()-begin)
    result=dict(status='PASS_OLD_INPUTS_FOR_CANONICAL_PAIRED_COVERAGE_CONTROL' if canonical_pair else 'PASS_ALL3000_OLD_INPUTS_AND_ALL2592_TRAIN_OBJECTIVES_PARITY',old_queries_tested=tested,old_ranking_eligible=old_eligible,full_ranking_eligible=ad['ranking_eligible'],
        batch_fields_bitwise_equal=True,native_candidate_IDs_equal=True,all_trained_supervision_masks_and_aliases_equal=not bool(training_differences),canonical_paired300_required=canonical_pair,documented_certification_differences=label_differences,training_certification_differences=training_differences,initialization_and_losses='Unchanged frozen CandidatePredictor and GroupedHeads; original .001 AdamW/.0001 decay,clip1,rank+0.5cross prediction',
        CPU_joint_step_seconds_median=statistics.median(times),CPU_joint_step_seconds_max=max(times),probe_steps_per_arm=len(selected),all_probe_weights_discarded=True,
        scientific_training_updates=0,GPU_wall_hours=0,DEV_read=False,preparation_sha256=sha(preparation_path),source_sha256=sha(__file__),old_protocol_sha256=sha(old_protocol_path),wall_seconds=time.monotonic()-start)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--canonical-pair',action='store_true');a=p.parse_args();run(a.preparation,a.out,a.canonical_pair)
