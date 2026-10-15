"""Verify actually matched temporal controls and known/unknown changes."""
import argparse
import numpy as np
import torch
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['vggt','dinov2'],required=True);a=p.parse_args()
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');manifests=[];states=[];validations=[];curves=[]
    for number in ['0p0','0p1']:
        root=runtime/'models'/f'cycle_{a.backbone}_lambda{number}_seed20261015_v3_plus200_v1'
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['status']!='COMPLETE_MATCHED_REAL_CYCLE_CONTINUATION':raise ValueError('temporal control incomplete')
        for name,digest in manifest['artifacts'].items():
            if sha256(root/name)!=digest:raise ValueError('temporal output changed')
        manifests.append(manifest);states.append(torch.load(root/'checkpoint.pt',map_location='cpu'))
        validations.append(json.loads((root/'validation.json').read_text())['rows']);curves.append(json.loads((root/'loss_curve.json').read_text()))
    for field in ['source_checkpoint_sha256','ordinary_minibatch_draw_sha256','cycle_query_schedule_sha256','additional_updates','total_updates','supervised_fit_queries','cycle_queries_with_visible_loss']:
        if manifests[0][field]!=manifests[1][field]:raise ValueError('temporal controls differ: '+field)
    first,last=states
    if sha256(manifests[0]['source_checkpoint'])!=manifests[0]['source_checkpoint_sha256']:
        raise ValueError('temporal source checkpoint changed')
    if any(len(v)!=300 for v in validations) or any(len(c)!=200 for c in curves):
        raise ValueError('temporal validation or actual update schedule is truncated')
    if len(first['cuda_rng_states'])!=len(last['cuda_rng_states']):
        raise ValueError('temporal CUDA RNG device count differs')
    source=torch.load(manifests[0]['source_checkpoint'],map_location='cpu')
    training_seed=source['settings']['training_seed'];native_seed=source['settings']['actual_native_inference_seed']
    if any(s['settings']['training_seed']!=training_seed for s in states):raise ValueError('temporal source/training seed differs')
    nr=[s['numpy_rng_state'] for s in states]
    if nr[0][0]!=nr[1][0] or not np.array_equal(nr[0][1],nr[1][1]) or nr[0][2:]!=nr[1][2:]:raise ValueError('temporal native-draw RNG differs')
    if not torch.equal(first['torch_rng_state'],last['torch_rng_state']) or any(not torch.equal(x,y) for x,y in zip(first['cuda_rng_states'],last['cuda_rng_states'])):raise ValueError('temporal dropout RNG consumption differs')
    for field in ['association_loss','cycle_loss']:
        if curves[0][0][field]!=curves[1][0][field]:raise ValueError('identical initial state/forward does not match')
    for left,right in zip(curves[0],curves[1]):
        if any(left[k]!=right[k] for k in ('additional_step','total_updates','cycle_key','visible_rows')):raise ValueError('actual temporal schedule differs')
    good=lambda s:s in ('correct_link','correct_null_within_recorded_history')
    bad=lambda s:s in ('wrong_link','missed_known_positive')
    changes=[];known_fixes=[];known_harms=[];uncertified_to_correct=[];correct_to_uncertified=[]
    for left,right in zip(*validations):
        if any(left[k]!=right[k] for k in ('key','query_gt_known','top8_positive','known_unmatched','actual_frozen_gmt_primary_state','original_native_top1_state')):raise ValueError('temporal validation labels/inputs differ')
        if left['state']!=right['state'] or left['prediction_index']!=right['prediction_index']:
            changes.append(dict(key=left['key'],without_cycle=left['state'],with_cycle=right['state'],native_state=left['original_native_top1_state'],actual_native_primary=left['actual_frozen_gmt_primary_state']))
        if bad(left['state']) and good(right['state']):known_fixes.append(left['key'])
        if good(left['state']) and bad(right['state']):known_harms.append(left['key'])
        if not good(left['state']) and not bad(left['state']) and good(right['state']):uncertified_to_correct.append(left['key'])
        if good(left['state']) and not good(right['state']) and not bad(right['state']):correct_to_uncertified.append(left['key'])
    proof=dict(status='PASS_IDENTICAL_SOURCE_AND_MATCHED_TEMPORAL_CONTROLS',backbone=a.backbone,source_checkpoint_sha256=manifests[0]['source_checkpoint_sha256'],
        actual_training_seed_from_executed_source_and_checkpoint=training_seed,actual_native_inference_seed=native_seed,
        raw_generic_provenance_seeds=[m['provenance']['seed'] for m in manifests],
        seed_reporting_correction='Pinned0eb9af7f fitter used generic provenance() default native seed14 even though seed_all(settings.training_seed) and restored source RNG use15. Raw artifacts remain unchanged; this audit resolves the actual seed. Future main-branch fitter writes the explicit training seed.',
        initial_association_and_cycle_loss_exact=True,final_numpy_and_Torch_dropout_RNG_exact=True,
        native_minibatch_draw_sha256=manifests[0]['ordinary_minibatch_draw_sha256'],cycle_schedule_sha256=manifests[0]['cycle_query_schedule_sha256'],
        additional_updates=200,total_updates=1200,visible_cycle_draws=manifests[0]['cycle_queries_with_visible_loss'],
        lambda0=manifests[0]['validation_metrics'],lambda0p1=manifests[1]['validation_metrics'],
        lambda0_shuffle=manifests[0]['query_shuffle_metrics'],lambda0p1_shuffle=manifests[1]['query_shuffle_metrics'],
        known_ranking_fix_keys=known_fixes,known_ranking_harm_keys=known_harms,uncertified_to_correct_keys=uncertified_to_correct,
        correct_to_uncertified_keys=correct_to_uncertified,all_changed_queries=changes,
        average_observed_cycle_loss=[float(np.mean([r['cycle_loss'] for r in curve])) for curve in curves],
        conclusion_scope='One primary-seed fixed300 ranking comparison. Unknown/ambiguous transitions are not certified wrong-link repairs; no actual learned Hungarian/Bank/final tracking correction or scientific-success claim.',provenance=provenance())
    save_json(ROOT/'results'/f'trackvggt_cycle_{a.backbone}_paired_v1.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('provenance','lambda0','lambda0p1','lambda0_shuffle','lambda0p1_shuffle','all_changed_queries')}))


if __name__=='__main__':main()
