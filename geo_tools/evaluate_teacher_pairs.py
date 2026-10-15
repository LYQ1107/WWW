"""Fixed physical-query teacher ranks, NULL/unknown coverage and paired states."""
import argparse,collections,hashlib,time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import *
from trackvggt_data import TrackEventData,association_losses,NEW


def predicted_state(order,target):
    if not target['query_gt_known']:return 'unknown_query'
    first=int(order[0]);null=len(target['positive'])-1
    if first==null:
        if target['known_unmatched']:return 'correct_null_within_recorded_history'
        return 'missed_known_positive' if target['top8_positive'] else 'unknown_candidate_coverage'
    if first in target['ambiguous_indices']:return 'ambiguous_reference'
    return 'correct_link' if bool(target['positive'][first]) else 'wrong_link'


def evaluate(model,data,indices,device='cpu',scorer='learned',shuffle=False,seed=20261015):
    if model is not None:model.eval()
    rows=[];losses=[];labels=[];pair_scores=[];weights=[];started=time.perf_counter()
    with torch.no_grad():
        for index in indices:
            features,target=data.get(index,device)
            permutation=None
            if shuffle:
                digest=int(hashlib.sha256(json.dumps(target['key']).encode()).hexdigest()[:8],16)
                permutation=torch.randperm(len(features['histories']),generator=torch.Generator().manual_seed(seed+digest)).tolist()
            if scorer=='native':logits=torch.cat((features['native_scores'],features['native_scores'].new_zeros(1)));aux=dict(neutral=False,layer_weights=[])
            elif scorer=='prior':logits=data.native_prior_scores(index).to(device);aux=dict(neutral=False,layer_weights=[])
            else:logits,aux=model(features,soft=False,history_permutation=permutation)
            if not torch.isfinite(logits).all():raise ValueError('nonfinite evaluated teacher scores')
            order=torch.argsort(logits,descending=True,stable=True)
            pos=target['positive'].clone();matched=pos.clone();matched[-1]=False
            rank=int(torch.where(matched[order])[0][0])+1 if matched.any() else None
            candidate_order=order[order<len(logits)-1]
            candidate_rank=int(torch.where(matched[candidate_order])[0][0])+1 if matched.any() else None
            state=predicted_state(order,target)
            loss=association_losses(logits,target)
            if loss is not None:losses.append(float(loss[0]))
            distribution=logits.float().softmax(0)
            for j in range(len(logits)-1):
                if target['query_gt_known'] and bool(target['known'][j]):labels.append(int(target['positive'][j]));pair_scores.append(float(distribution[j]))
            current_weights=[value.detach().cpu().tolist() for value in aux['layer_weights'] if value is not None]
            weights.extend(current_weights)
            rows.append(dict(key=target['key'],scene=target['scene'],patch_bin=target['patch_bin'],query_gt_known=target['query_gt_known'],
                any_history_positive=target['any_history_positive'],top8_positive=target['top8_positive'],known_unmatched=target['known_unmatched'],
                prediction_index=int(order[0]),state=state,positive_first_rank_including_null=rank,matched_candidate_first_rank=candidate_rank,
                logits=logits.cpu().tolist(),neutral_small_target=aux['neutral'],layer_weights=current_weights,query_permutation=permutation,
                original_native_top1_state=target['native_top1_state'],actual_frozen_gmt_primary_state=target['actual_primary_state']))
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    matched=[row for row in rows if row['top8_positive']];known=[row for row in rows if row['query_gt_known']]
    ranks=[row['positive_first_rank_including_null'] for row in matched];candidate_ranks=[row['matched_candidate_first_rank'] for row in matched]
    has_pairs=len(set(labels))==2
    metrics=dict(requests=len(rows),known_queries=len(known),matched_eligible=len(matched),
        correct_identity_any_history=sum(row['any_history_positive'] for row in rows),positive_lost_by_top8=sum(row['any_history_positive'] and not row['top8_positive'] for row in rows),
        matched_query_coverage=len(matched)/len(rows) if rows else None,
        Recall1=sum(rank==1 for rank in ranks)/len(ranks) if ranks else None,Recall5=sum(rank<=5 for rank in ranks)/len(ranks) if ranks else None,
        MRR=sum(1/rank for rank in ranks)/len(ranks) if ranks else None,
        matched_candidate_Recall1_excluding_null=sum(rank==1 for rank in candidate_ranks)/len(candidate_ranks) if candidate_ranks else None,
        all_request_known_positive_success1=sum(rank==1 for rank in ranks)/len(rows) if rows else None,
        pair_auc=float(roc_auc_score(labels,pair_scores)) if has_pairs else None,pair_ap=float(average_precision_score(labels,pair_scores)) if has_pairs else None,
        supervised_loss=float(np.mean(losses)) if losses else None,states=dict(collections.Counter(row['state'] for row in rows)),
        neutral_small_target_queries=sum(row['neutral_small_target'] for row in rows),
        layer_weight_mean=np.mean(weights,axis=0).tolist() if weights else None,layer_weight_std=np.std(weights,axis=0).tolist() if weights else None)
    return dict(metrics=metrics,rows=rows,seconds=time.perf_counter()-started,
        timing_scope='cached-target pair diagnostic including array/history loading and model scoring; not image/encoder/full-tracker deployment FPS')


def build_model(settings):
    from gtr.geometry.track_ag_teacher import TrackRelationTeacher,NativeRelationMLP
    if settings['architecture']=='native_mlp':return NativeRelationMLP(settings['native_dim'],settings.get('native_mlp_capacity','qd'))
    return TrackRelationTeacher(settings['native_dim'],settings['channels'],architecture=settings['architecture'],ag_mode=settings['ag_mode'],
        qd_mode=settings['qd_mode'],k=settings['k'],summary=settings['summary'],matched_dino=settings['backbone']=='dinov2')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--scorer',choices=['learned','native','prior'],default='learned')
    p.add_argument('--backbone',choices=['vggt','dinov2'],default='vggt');p.add_argument('--split',choices=['fit','validation'],default='validation')
    p.add_argument('--output',required=True);p.add_argument('--query-shuffle',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--cohort');a=p.parse_args()
    path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    model=None;settings=None
    if a.scorer=='learned':
        checkpoint=torch.load(a.checkpoint,map_location='cpu');settings=checkpoint['settings'];a.backbone=settings['backbone']
        if a.cohort is None:a.cohort=checkpoint.get('cohort_path')
        model=build_model(settings);model.load_state_dict(checkpoint['model'],strict=True);model.to(a.device)
    data=TrackEventData(a.backbone,a.cohort)
    if settings and settings.get('feature_plan'):
        from trackvggt_crop_data import CropEventData
        if sha256(settings['feature_plan'])!=settings['feature_plan_sha256']:raise ValueError('checkpoint crop scope changed')
        data=CropEventData(data,settings['feature_plan'],settings['feature_variant'])
    indices=getattr(data,a.split)
    if a.scorer=='learned' and checkpoint['cohort_sha256']!=sha256(data.cohort_path):raise ValueError('evaluated checkpoint/cohort mismatch')
    result=evaluate(model,data,indices,a.device,a.scorer,a.query_shuffle)
    result.update(status='COMPLETE_FIXED_EVENT_RANKING_DIAGNOSTIC',scorer=a.scorer,split=a.split,backbone=a.backbone,query_shuffle=a.query_shuffle,
        model_settings=settings,checkpoint=a.checkpoint,checkpoint_sha256=sha256(a.checkpoint) if a.checkpoint else None,
        source_artifacts=data.source_artifacts,scope='post-run known-pure labels, one fixed native-history cohort; ambiguous references and missing positives retained; not actual learned Hungarian/final tracking improvement',provenance=provenance())
    save_json(path,result);print(json.dumps(result['metrics']))


if __name__=='__main__':main()
