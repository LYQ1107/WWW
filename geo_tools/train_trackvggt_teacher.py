"""Finite TRAIN fitting of AG alone, then separately initialized QD controls."""
import argparse,time
import numpy as np
import torch
from common import *
from trackvggt_data import TrackEventData,association_losses,NEW
from evaluate_teacher_pairs import evaluate,build_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--backbone',choices=['vggt','dinov2'],default='vggt')
    p.add_argument('--architecture',choices=['ag','qd','native_mlp'],default='ag');p.add_argument('--ag-mode',choices=['roi','A','B','C'],default='C')
    p.add_argument('--qd-mode',choices=['last','mean','concat','global','track'],default='track');p.add_argument('--k',type=int,choices=[16,24,32],default=24)
    p.add_argument('--summary',choices=['mean','ema'],default='mean');p.add_argument('--seed',type=int,default=20261015)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--ag-init');p.add_argument('--resume');p.add_argument('--device',default='cuda')
    p.add_argument('--native-mlp-capacity',choices=['ag','qd'],default='ag')
    p.add_argument('--cohort',default=None)
    p.add_argument('--feature-plan',type=Path);p.add_argument('--feature-variant',choices=['full','local','combined'],default='full')
    a=p.parse_args();out=NEW/'models'/a.tag
    if out.exists():raise FileExistsError(out)
    if a.steps not in (150,1000):raise ValueError('prespecified pilot150 or full1000 steps only')
    if a.architecture=='qd' and not a.ag_init:raise ValueError('formal QD requires the separately fitted AG source, no simultaneous Phase2 shortcut')
    out.mkdir(parents=True);seed_all(a.seed)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cudnn.deterministic=True
    started=time.perf_counter();data=TrackEventData(a.backbone,a.cohort)
    if a.feature_plan:
        from trackvggt_crop_data import CropEventData
        data=CropEventData(data,a.feature_plan,a.feature_variant)
    elif a.feature_variant!='full':raise ValueError('crop features require the fixed comparison plan')
    settings=dict(backbone=a.backbone,native_dim=data.native_dim,channels=data.channels,architecture=a.architecture,ag_mode=a.ag_mode,
        qd_mode=a.qd_mode,k=a.k,summary=a.summary,native_mlp_capacity=a.native_mlp_capacity,training_seed=a.seed,actual_native_inference_seed=20261014,
        optimizer='AdamW',lr=1e-4,weight_decay=.01,clip_norm=1.,batch_size=a.batch_size,target_steps=a.steps,soft_relevance_warmup_steps=150)
    settings.update(feature_variant=a.feature_variant,feature_plan=str(a.feature_plan.resolve()) if a.feature_plan else None,
        feature_plan_sha256=sha256(a.feature_plan) if a.feature_plan else None)
    model=build_model(settings).to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    if a.ag_init:
        source=torch.load(a.ag_init,map_location='cpu')
        if source['settings']['backbone']!=a.backbone or source['settings']['ag_mode']!='C':raise ValueError('QD source must be a matching fitted learned AG-C')
        for key in ('native_dim','channels','summary'):
            if source['settings'][key]!=settings[key]:raise ValueError('QD source input/query contract mismatch: '+key)
        if source['cohort_sha256']!=sha256(data.cohort_path):raise ValueError('QD source changed locked event cohort')
        for key in ('feature_variant','feature_plan_sha256'):
            if source['settings'].get(key,'full' if key=='feature_variant' else None)!=settings[key]:raise ValueError('QD source feature scope mismatch')
        model.query.load_state_dict({k.removeprefix('query.'):v for k,v in source['model'].items() if k.startswith('query.')},strict=True)
        model.sampler.load_state_dict({k.removeprefix('sampler.'):v for k,v in source['model'].items() if k.startswith('sampler.')},strict=True)
    start_step=0
    if a.resume:
        source=torch.load(a.resume,map_location='cpu')
        if source['cohort_sha256']!=sha256(data.cohort_path):raise ValueError('resume changed locked event cohort')
        for key in ('backbone','native_dim','channels','architecture','ag_mode','qd_mode','k','summary','native_mlp_capacity','training_seed'):
            if source['settings'][key]!=settings[key]:raise ValueError('resume changed model/seed/setting')
        for key in ('feature_variant','feature_plan_sha256'):
            if source['settings'].get(key,'full' if key=='feature_variant' else None)!=settings[key]:raise ValueError('resume changed feature scope')
        model.load_state_dict(source['model'],strict=True);optimizer.load_state_dict(source['optimizer']);start_step=source['completed_steps']
        torch.set_rng_state(source['torch_rng_state']);np.random.set_state(source['numpy_rng_state'])
        if a.device.startswith('cuda'):torch.cuda.set_rng_state_all(source['cuda_rng_states'])
    eligible=[]
    for index in data.fit:
        _,_,event=data.event(index);row=data.rows[tuple(data.items[index]['key'])]
        feature_valid=bool(event['candidate_tracks']) and (bool(a.feature_plan) or a.architecture=='native_mlp' or a.ag_mode=='roi' or data.effective_patch_count(index)>0)
        if event['loss_eligible'] and feature_valid:eligible.append(index)
    if not eligible:raise ValueError('no legal supervised fit relations; no fabricated samples')
    setup_seconds=time.perf_counter()-started;training_started=time.perf_counter();curve=[];neutral_fit_forwards=0;zero_gradient_batches=0
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for step in range(start_step,a.steps):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[];parts=[]
        selected=np.random.choice(eligible,size=a.batch_size,replace=len(eligible)<a.batch_size).tolist()
        for index in selected:
            features,target=data.get(index,a.device)
            logits,aux=model(features,soft=a.ag_mode=='C' and a.architecture!='native_mlp' and not a.ag_init and step<150)
            value=association_losses(logits,target)
            if value is None or (aux['neutral'] and not a.feature_plan):raise ValueError('supervised event selection/model validity mismatch')
            neutral_fit_forwards+=int(aux['neutral'])
            losses.append(value[0]);parts.append([float(value[1].detach()),float(value[2].detach())])
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        if not loss.requires_grad:
            # Matched crop variants draw the same actual queries, including
            # full-image zero-center native fallbacks. A constant batch has
            # exactly zero learned gradient; no fabricated token/history path.
            loss=loss+sum(parameter.sum()*0 for parameter in model.parameters());zero_gradient_batches+=1
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm) or any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):raise ValueError('nonfinite gradient')
        optimizer.step()
        if (step+1)%25==0 or step==start_step:
            row=dict(step=step+1,loss=float(loss.detach()),association_loss=float(np.mean([v[0] for v in parts])),rank_loss=float(np.mean([v[1] for v in parts])),gradient_norm_before_clip=float(norm))
            curve.append(row);print(json.dumps(row),flush=True)
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    training_seconds=time.perf_counter()-training_started
    checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),settings=settings,completed_steps=a.steps,
        cohort_path=str(data.cohort_path),cohort_sha256=sha256(data.cohort_path),source_artifacts=data.source_artifacts,torch_rng_state=torch.get_rng_state(),numpy_rng_state=np.random.get_state(),
        cuda_rng_states=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [],provenance=provenance(a.seed))
    torch.save(checkpoint,out/'checkpoint.pt');save_json(out/'loss_curve.json',curve)
    validation=evaluate(model,data,data.validation,a.device)
    save_json(out/'validation.json',validation)
    shuffled=evaluate(model,data,data.validation,a.device,shuffle=True,seed=20261015)
    save_json(out/'validation_query_shuffle.json',shuffled)
    report=dict(status='COMPLETE_FINITE_TEACHER_FIT_AND_FIXED_VALIDATION',settings=settings,completed_steps=a.steps,starting_step=start_step,
        supervised_fit_queries=len(eligible),all_fit_queries=len(data.fit),validation_queries=len(data.validation),
        neutral_fit_query_forwards=neutral_fit_forwards,zero_gradient_batches=zero_gradient_batches,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),setup_seconds=setup_seconds,training_seconds=training_seconds,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None,
        validation_metrics=validation['metrics'],query_shuffle_metrics=shuffled['metrics'],
        query_shuffle_seed=20261015,empty_candidate_queries_excluded_from_gradient_fit=sum(not data.event(i)[2]['candidate_tracks'] for i in data.fit),
        learned_ag_score_weights=model.sampler.mix.detach().softmax(0).cpu().tolist() if hasattr(model,'sampler') else None,
        ag_init=a.ag_init,ag_init_sha256=sha256(a.ag_init) if a.ag_init else None,resume=a.resume,resume_sha256=sha256(a.resume) if a.resume else None,
        checkpoint=str(out/'checkpoint.pt'),checkpoint_sha256=sha256(out/'checkpoint.pt'),source_artifacts=data.source_artifacts,
        artifacts={file.name:sha256(file) for file in out.iterdir() if file.is_file()},
        frozen_backbones_loaded_during_fit=False,backbone_extraction_cost_accounted_separately=True,
        scope='finite small-module TRAIN fit; locked candidate histories, pure known-label listwise+ranking, frozen target cache; validation and shuffle are pair diagnostics, not closed-loop tracking/TEACHER_SUCCESS',provenance=provenance(a.seed))
    save_json(out/'manifest.json',report);print(json.dumps(dict(status=report['status'],steps=a.steps,parameters=report['trainable_parameters'],validation=validation['metrics'])),flush=True)


if __name__=='__main__':main()
