"""Equal200-update QD continuations, lambda0/.1 from the identical fitted state.

Both controls execute the same real cycle forwards (also lambda0) so their
dropout RNG consumption and native minibatch/query schedule stay matched.
"""
import argparse
import time
import numpy as np
import torch
from common import *
from trackvggt_data import TrackEventData,association_losses
from trackvggt_cycle_features import NativeCycleData,forward_cycle
from evaluate_teacher_pairs import build_model,evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--lambda-cycle',type=float,choices=[0.,.1],required=True)
    p.add_argument('--tag',required=True);a=p.parse_args();runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');out=runtime/'models'/a.tag
    if out.exists():raise FileExistsError(out)
    state=torch.load(a.source,map_location='cpu');settings=state['settings']
    if settings['architecture']!='qd' or settings['qd_mode']!='track' or state['completed_steps']!=1000:raise ValueError('fixed completed no-cycle trackQD1000 source required')
    if settings.get('feature_plan') or settings.get('feature_variant','full')!='full':raise ValueError('this bounded cycle experiment uses original full-image V3 teacher')
    out.mkdir(parents=True);seed_all(settings['training_seed']);started=time.perf_counter()
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cudnn.deterministic=True
    data=TrackEventData(settings['backbone'],state['cohort_path'])
    if sha256(data.cohort_path)!=state['cohort_sha256'] or data.cohort.get('cohort_version')!=3:raise ValueError('cycle source cohort differs')
    cycles=NativeCycleData(data);model=build_model(settings).to('cuda');model.load_state_dict(state['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);optimizer.load_state_dict(state['optimizer'])
    eligible=[i for i in data.fit if data.event(i)[2]['candidate_tracks'] and data.event(i)[2]['loss_eligible'] and data.effective_patch_count(i)>0]
    if not eligible:raise ValueError('no actual supervised QD population')
    torch.set_rng_state(state['torch_rng_state']);np.random.set_state(state['numpy_rng_state']);torch.cuda.set_rng_state_all(state['cuda_rng_states'])
    setup=time.perf_counter()-started;started=time.perf_counter();curve=[];draw_hash=hashlib.sha256();cycle_keys=[];torch.cuda.reset_peak_memory_stats()
    for step in range(200):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        selected=np.random.choice(eligible,size=4,replace=len(eligible)<4).tolist();draw_hash.update(json.dumps(selected).encode())
        for index in selected:
            inputs,target=data.get(index,'cuda');logits,aux=model(inputs,soft=False);value=association_losses(logits,target)
            if value is None or aux['neutral']:raise ValueError('unchanged QD fit population validity')
            losses.append(value[0])
        association=torch.stack(losses).mean()
        ci=step%len(cycles.specs);features,masks=cycles.get(ci,'cuda');cycle,detail=forward_cycle(model,features,masks)
        cycle_keys.append(cycles.specs[ci]['key']);loss=association+a.lambda_cycle*cycle
        if not torch.isfinite(loss):raise ValueError('nonfinite actual cycle continuation')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('nonfinite cycle gradient')
        optimizer.step();row=dict(additional_step=step+1,total_updates=1001+step,loss=float(loss.detach()),association_loss=float(association.detach()),
            cycle_loss=float(cycle.detach()),cycle_key=cycle_keys[-1],visible_rows=detail['visible_rows'],gradient_norm_before_clip=float(norm))
        curve.append(row)
        if step==0 or (step+1)%25==0:print(json.dumps(row),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),settings=dict(settings,lambda_cycle=a.lambda_cycle),
        completed_steps=1200,cycle_additional_steps=200,cohort_path=str(data.cohort_path),cohort_sha256=sha256(data.cohort_path),
        source_checkpoint=str(a.source),source_checkpoint_sha256=sha256(a.source),source_artifacts=data.source_artifacts,
        torch_rng_state=torch.get_rng_state(),numpy_rng_state=np.random.get_state(),cuda_rng_states=torch.cuda.get_rng_state_all(),provenance=provenance(settings['training_seed']))
    torch.save(checkpoint,out/'checkpoint.pt');save_json(out/'loss_curve.json',curve)
    validation=evaluate(model,data,data.validation,'cuda');shuffled=evaluate(model,data,data.validation,'cuda',shuffle=True)
    save_json(out/'validation.json',validation);save_json(out/'validation_query_shuffle.json',shuffled)
    proof=dict(status='COMPLETE_MATCHED_REAL_CYCLE_CONTINUATION',lambda_cycle=a.lambda_cycle,additional_updates=200,total_updates=1200,
        source_checkpoint=str(a.source),source_checkpoint_sha256=sha256(a.source),cycle_scope='locked128 observation-only query subset; B<=4; A_current one actual selected detection plus NULL',
        ordinary_minibatch_draw_sha256=draw_hash.hexdigest(),cycle_query_schedule_sha256=hashlib.sha256(json.dumps(cycle_keys).encode()).hexdigest(),
        cycle_forward_also_executed_for_lambda0=True,cycle_queries_with_visible_loss=sum(row['visible_rows']>0 for row in curve),
        supervised_fit_queries=len(eligible),validation_metrics=validation['metrics'],query_shuffle_metrics=shuffled['metrics'],
        setup_seconds=setup,training_seconds=seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_sha256=sha256(out/'checkpoint.pt'),artifacts={p.name:sha256(p) for p in out.iterdir() if p.is_file()},
        training_seed=settings['training_seed'],actual_native_inference_seed=settings['actual_native_inference_seed'],
        no_new_DEV12_release_or_future_inputs=True,scientific_success=False,provenance=provenance(settings['training_seed']))
    save_json(out/'manifest.json',proof);print(json.dumps({k:v for k,v in proof.items() if k not in ('artifacts','provenance')}))


if __name__=='__main__':main()
