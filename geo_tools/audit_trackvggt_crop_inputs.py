"""Real all-query full/local/combined native/label equality before fitting."""
import torch
from common import *
from trackvggt_data import TrackEventData
from trackvggt_crop_data import CropEventData


def main():
    plan=ROOT/'manifests/trackvggt_crop_comparison_v1.json';value=json.loads(plan.read_text());reports=[]
    for backbone in ['vggt','dinov2']:
        base=TrackEventData(backbone,value['base_cohort']);variants={name:CropEventData(base,plan,name) for name in value['comparison']}
        data=variants['full'];improved=dict(fit=0,validation=0);eligible={}
        for index in data.fit+data.validation:
            ordinary,target=data.get(index)
            if data.base.effective_patch_count(index)==0 and variants['local'].effective_patch_count(index)>0:improved['fit' if index in data.fit else 'validation']+=1
            for name in ['local','combined']:
                features,labels=variants[name].get(index)
                for field in ['current','pair','native_scores']:
                    if not torch.equal(ordinary[field],features[field]):raise ValueError('crop changed native actor evidence')
                for old,new in zip(ordinary['histories'],features['histories']):
                    for field in ['features','times','cameras']:
                        if not torch.equal(old[field],new[field]):raise ValueError('crop changed native history')
                    if old['observed_length']!=new['observed_length']:raise ValueError('crop changed native length')
                for field in target:
                    if isinstance(target[field],torch.Tensor):equal=torch.equal(target[field],labels[field])
                    else:equal=target[field]==labels[field]
                    if not equal:raise ValueError('crop changed label/query membership')
        for name,dataset in variants.items():
            eligible[name]=[i for i in dataset.fit if dataset.event(i)[2]['loss_eligible'] and dataset.event(i)[2]['candidate_tracks']]
        if any(eligible[name]!=eligible['full'] for name in eligible):raise ValueError('crop matched minibatch populations differ')
        reports.append(dict(backbone=backbone,fit=len(data.fit),validation=len(data.validation),matched_supervised_fit_queries=len(eligible['full']),
            formerly_zero_center_now_observable=improved,all_native_features_histories_pairs_and_labels_exact=True,
            feature_manifests={path:sha256(path) for path in value['feature_manifests'].values()}))
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');queues=[]
    for plan_id,gpu in [('trackvggt_crop_prechecks_v1',6),('trackvggt_crop_tokens_v1',4),('trackvggt_crop_tokens_v1',6)]:
        path=runtime/'queues'/f'{plan_id}_gpu{gpu}/manifest.json';q=json.loads(path.read_text())
        if q['status']!='FINISHED':raise ValueError('crop stage incomplete')
        queues.append(dict(path=str(path),sha256=sha256(path),seconds=sum(s['seconds'] for s in q['stages'])))
    seconds=sum(q['seconds'] for q in queues)
    if seconds>3600:raise ValueError('crop encoder wall budget exceeded')
    token_bytes=sum(p.stat().st_size for p in (runtime/'tokens').rglob('*.npz'))
    if token_bytes>2*(1<<30):raise ValueError('shared target-pool cap exceeded')
    proof=dict(status='PASS_ALL_FIXED_CROP_INPUTS',plan_sha256=sha256(plan),rows=reports,queues=queues,
        new_frozen_crop_GPU_wall_hours_including_probes=seconds/3600,all_shared_NPZ_bytes=token_bytes,maximum_shared_NPZ_bytes=2*(1<<30),
        matched_queries_seeds_update_budget_required=True,no_GT_generated_crop_or_history=True,teacher_benefit='NOT_MEASURED',provenance=provenance())
    save_json(ROOT/'results/trackvggt_crop_inputs_complete_v1.json',proof);print(json.dumps({k:v for k,v in proof.items() if k not in ('rows','queues','provenance')}))


if __name__=='__main__':main()
