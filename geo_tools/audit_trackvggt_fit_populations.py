"""Disclose actual gradient-eligible populations and reused upstream updates.

No new data selection, fit, validation selection or model update. Strong native
controls share the full task/cohort but do not have missing visual-token support.
"""
import collections
import textwrap
from common import *
from trackvggt_data import TrackEventData


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');cohort=runtime/'cohorts/hard_events_expanded_v3/cohort.json'
    profiles=[];reference={}
    for backbone in ['vggt','dinov2']:
        data=TrackEventData(backbone,cohort)
        for mode in ['visual_AG_QD','ROI_or_native']:
            counts=collections.Counter();keys=[]
            for index in data.fit:
                item,_,event=data.event(index)
                if data.effective_patch_count(index)!=data.rows[tuple(item['key'])]['raw_patch_centers']:
                    raise ValueError('current effective support differs from the legacy full-image support field')
                if not event['loss_eligible']:counts['masked_no_supervision']+=1;continue
                if not event['candidate_tracks']:counts['no_candidates_constant_NULL']+=1;continue
                if mode=='visual_AG_QD' and data.effective_patch_count(index)==0:
                    counts['no_raw_patch_center_no_visual_gradient']+=1;continue
                keys.append(item['key']);counts['eligible']+=1
                has_positive=any(track in event['positive_tracks'] for track in event['candidate_tracks'])
                null_positive=event['known_unmatched_within_all_recorded_history'] and not has_positive
                if not has_positive and not null_positive:raise ValueError('supervised event has no known positive/NULL target')
                counts['eligible_genuine_NULL' if null_positive else 'eligible_positive']+=1
            digest=hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()
            if mode in reference and reference[mode]!=digest:raise ValueError('DINO/VGGT effective training populations differ')
            reference[mode]=digest
            profiles.append(dict(backbone=backbone,mode=mode,full_fit_queries=len(data.fit),counts=dict(counts),
                ordered_eligible_keys_sha256=digest,cohort_sha256=sha256(cohort)))
    runs=[]
    for tag in ['ag_vggt_C_mean_k24_seed20261015_v3_1000_v1','ag_vggt_roi_mean_k24_seed20261015_v3_1000_v1',
        'ag_dinov2_C_mean_k24_seed20261015_v3_1000_v1','native_mlp_vggt_C_mean_k24_seed20261015_v3_1000_v1',
        'native_mlp_qd_capacity_seed20261015_v3_1000_v1','qd_vggt_track_k24_seed20261015_v3_1000_v1',
        'qd_dinov2_track_k24_seed20261015_v3_1000_v1']:
        path=runtime/'models'/tag/'manifest.json';m=json.loads(path.read_text());settings=m['settings']
        source_root=Path(m['provenance']['source_worktree'])
        for source in ['geo_tools/trackvggt_data.py','geo_tools/train_trackvggt_teacher.py']:
            if sha256(source_root/source)!=m['provenance']['source_hashes'][source]:raise ValueError('immutable executed fit source changed')
        source=(source_root/'geo_tools/train_trackvggt_teacher.py').read_text()
        block=textwrap.dedent(source[source.index('    eligible=[]'):source.index('    setup_seconds=')]).strip()
        conditions=["a.architecture=='native_mlp' or a.ag_mode=='roi' or row['raw_patch_centers']>0",
            "bool(a.feature_plan) or a.architecture=='native_mlp' or a.ag_mode=='roi' or data.effective_patch_count(index)>0"]
        template="""eligible=[]
for index in data.fit:
    _,_,event=data.event(index);row=data.rows[tuple(data.items[index]['key'])]
    feature_valid=bool(event['candidate_tracks']) and ({condition})
    if event['loss_eligible'] and feature_valid:eligible.append(index)
if not eligible:raise ValueError('no legal supervised fit relations; no fabricated samples')"""
        if block not in [template.format(condition=c) for c in conditions] or settings.get('feature_plan'):
            raise ValueError('unrecognized actual no-crop training eligibility block')
        mode='ROI_or_native' if settings['architecture']=='native_mlp' or settings['ag_mode']=='roi' else 'visual_AG_QD'
        profile=next(p for p in profiles if p['backbone']==settings['backbone'] and p['mode']==mode)
        if profile['counts']['eligible']!=m['supervised_fit_queries']:raise ValueError('actual fitted eligibility counter differs')
        upstream=0
        if m.get('ag_init'):
            if sha256(m['ag_init'])!=m['ag_init_sha256']:raise ValueError('reused AG initialization changed')
            upstream=json.loads((Path(m['ag_init']).parent/'manifest.json').read_text())['completed_steps']
        runs.append(dict(tag=tag,supervised_fit_queries=m['supervised_fit_queries'],formal_phase_total_updates=m['completed_steps'],
            reused_upstream_AG_updates=upstream,ordered_eligible_keys_sha256=profile['ordered_eligible_keys_sha256'],manifest_sha256=sha256(path),
            executed_eligibility_source=str(source_root/'geo_tools/train_trackvggt_teacher.py'),executed_eligibility_block=block,
            source_equivalence='Legacy direct raw_patch_centers and later effective_patch_count are equal for these full-image no-crop inputs; native/ROI exception and loss/candidate eligibility unchanged.'))
    crop=json.loads((ROOT/'results/trackvggt_crop_primary_results_v1.json').read_text())
    if len(crop['rows'])!=6 or any(r['fit_queries']!=256 or r['supervised_fit_queries']!=234 for r in crop['rows']):raise ValueError('crop matched population differs')
    proof=dict(status='COMPLETE_ACTUAL_FIT_POPULATION_DISCLOSURE',profiles=profiles,runs=runs,
        matched_crop_fit=256,matched_crop_supervised=234,model_updates=0,
        interpretation='Full native/ROI controls share the frozen3000 task and validation300 but use2733 effective supervised queries; full visual AG/QD uses2201. Thus they are strong task/capacity controls, not identical-gradient-population controls. Matched VGGT/DINO visual populations are exactly equal; all crop variants preserve the same234 supervised queries including full-image native fallbacks.',
        update_budget='All cited formal phases total1000 updates (pilot150 then850 resumed). QD additionally reuses a learned1000-update AG query/sampler initialization for both VGGT and DINO. NativeMLP has no AG pretraining; no claim of equal total upstream optimization or identical sampling across those distinct controls.',
        limitations='Eligibility differences limit capacity-only or intrinsic-feature attribution against native/ROI. Do not erase them by relabeling cohorts or retrospectively changing completed fits. Matched DINO/crop/query-shuffle/cycle evidence and overall gate remain separately audited.',
        provenance=provenance())
    save_json(ROOT/'results/trackvggt_fit_population_disclosure_v1.json',proof)
    print(json.dumps(dict(status=proof['status'],profiles=profiles)))


if __name__=='__main__':main()
