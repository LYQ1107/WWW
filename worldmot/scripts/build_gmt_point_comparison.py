#!/usr/bin/env python3
"""Publish partial official table without replacing missing scores by zero."""
import csv
import json
from pathlib import Path
import statistics
from worldmot.models.person_detector import sha256_file

HERE=Path(__file__).resolve().parents[1]
NAS=Path('/data1/liuyeqiang/nas/A100/WorldMOT')
FIELDS=['HOTA','AssA','DetA','IDF1','CVIDF1','MOTA','IDSW','Frag','CVMA']


def extract(data):
    return {**{k:v for k,v in data['TrackEval']['combined'].items() if k in FIELDS},
        **{k:v for k,v in data['VisionTrack_MATLAB']['combined'].items() if k in FIELDS}}


def main():
    output=HERE/'results';source=output/'gmt_vggt_original_official_baselines.json'
    original=json.loads(source.read_text())['methods'];rows=[]
    def row(name,detector,association,metrics=None,fps=None,**extra):
        rows.append({'Method':name,'Detector':detector,'Association':association,
            **{k:metrics.get(k,'NOT_AVAILABLE') if metrics else 'NOT_AVAILABLE' for k in FIELDS},
            'E2E_FPS':fps if fps is not None else 'NOT_AVAILABLE',
            'Status':'OFFICIAL_COMPLETE_14_VALID_GT_22_INFERENCE' if metrics else 'PENDING_TRAIN_VALIDATION_FREEZE_AND_FULL_TEST',**extra})
    native=original['gmt_native'];row('GMT Native','GMT','Native GTA',extract(native),native['real_e2e_fps'],
        Parameters='NOT_AVAILABLE_PENDING_ASSOCIATION_COUNT_AUDIT',Timing='full-video delivered throughput; per-frame latency differs')
    old=original['v1'];row('Original V1','RTMDet','Omega 3D rules',extract(old),old['real_e2e_fps'],Parameters=0,Timing=old['timing_scope'])
    seeds=[original[f'v2_joint_seed{x}'] for x in [20261022,20261023,20261024]]
    scores=[extract(x) for x in seeds];mean={k:statistics.mean(x[k] for x in scores) for k in FIELDS}
    row('Original V2 (3-seed mean)','RTMDet','PWQT',mean,statistics.mean(x['real_e2e_fps'] for x in seeds),
        Parameters='NOT_AVAILABLE_PENDING_ASSOCIATION_COUNT_AUDIT',Timing='mean of real full-video FPS; three seeds',
        standard_deviation={k:statistics.pstdev(x[k] for x in scores) for k in FIELDS})
    specs=[('GMTDet + 2D IoU/Motion','IoU/Motion',0,NAS/'final_comparison_20261023/2d_controls_v1','iou_motion','test','test_evaluation'),
        ('GMTDet + 2D Appearance/Motion','ROI/Motion',0,NAS/'final_comparison_20261023/2d_controls_v1','appearance_motion','test','test_evaluation'),
        ('GMTDet + V1','Omega 3D rules',0,NAS/'gmt_detector_comparison_20261023','gmt_v1_v2','test','evaluation'),
        ('GMTDet + V2','PWQT','NOT_AVAILABLE_PENDING_FIT',NAS/'gmt_detector_comparison_20261023','gmt_v2_joint_seed20261022','test','evaluation'),
        ('GMTDet + VGGT Temporal','Single-view points',0,NAS/'vggt_point_association_20261023','temporal','test_v1','test_evaluation_v1'),
        ('GMTDet + VGGT Joint','Multi-view points',0,NAS/'vggt_point_association_20261023','joint','test_v1','test_evaluation_v1')]
    for name,association,parameters,root,method,pred_parent,eval_parent in specs:
        if association in ['Omega 3D rules','PWQT']:
            pred=root/f'{pred_parent}_{method}';evaluation=root/f'{eval_parent}_{method}/metrics.json'
        else:pred=root/pred_parent/method;evaluation=root/eval_parent/method/'metrics.json'
        metrics=None;fps=None;timing='NOT_AVAILABLE; full TEST pending'
        if evaluation.exists():
            data=json.loads(evaluation.read_text());assert data['status']=='COMPLETE_VALID_SCOPE'
            assert not data['raw_GT_modified'] and not data['custom_metrics_used'] and data['box_source']=='tracks_2d_by_camera'
            manifests=[json.loads(p.read_text()) for p in pred.glob('scene_*/manifest.json')]
            assert len(manifests)==22 and all(m['status']=='COMPLETE' and m['full_video'] and not m['GT_in_inference'] for m in manifests)
            assert sum(m['frames'] for m in manifests)==29019
            metrics=extract(data)
            if all(m.get('real_e2e_seconds') is not None for m in manifests):fps=29019/sum(m['real_e2e_seconds'] for m in manifests)
            timing='real RGB-to-ID full-video throughput; GPU sharing recorded in manifests; not exclusive latency ranking'
        row(name,'GMT',association,metrics,fps,Parameters=parameters,Timing=timing)
    for r in rows:
        r['delta_HOTA_vs_GMT']=r['HOTA']-rows[0]['HOTA'] if isinstance(r['HOTA'],(int,float)) else 'NOT_AVAILABLE'
        r['delta_IDF1_vs_GMT']=r['IDF1']-rows[0]['IDF1'] if isinstance(r['IDF1'],(int,float)) else 'NOT_AVAILABLE'
    report={'status':'PARTIAL_NEW_METHOD_OFFICIAL_TEST_PENDING' if any(r['HOTA']=='NOT_AVAILABLE' for r in rows) else 'ALL_PRIMARY_OFFICIAL_SCORES_AVAILABLE',
        'source_original_baselines_sha256':sha256_file(source),'official_rows':rows,
        'raw_GT_modified':False,'official_valid_GT_scenes':14,'inference_required_scenes':22,
        'new_TEST_used_for_selection':False,
        'parameter_note':'Zero means no association parameters trained in this experiment, not a zero-parameter detector/backbone.',
        'pilot_speed_excluded_from_main_table':'Short TRAIN prefix speed/quality reports are separate artifacts.',
        'remaining':'Matched-budget V2 fit and all new frozen full TEST, per-scene attribution and exclusive timing audit'}
    (output/'gmt_vggt_point_final_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    columns=['Method','Detector','Association','Parameters',*FIELDS,'E2E_FPS','delta_HOTA_vs_GMT','delta_IDF1_vs_GMT','Status','Timing']
    with (output/'gmt_vggt_point_final_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    fps_rows=[]
    for name,data in original.items():
        fps_rows.append({'scope':'COMPLETE_22_TEST_VIDEOS','method':name,'stage':'full_video_delivery',
            'timestamps':29019,'real_e2e_FPS':data['real_e2e_fps'],'VGGT_model_only_FPS':'NOT_AVAILABLE',
            'P50_s':'NOT_AVAILABLE_PENDING_RAW_TIMING_AGGREGATION','P95_s':'NOT_AVAILABLE_PENDING_RAW_TIMING_AGGREGATION',
            'conditions':data['timing_scope'],'source':data['source_metrics']['path']})
    point=NAS/'vggt_point_association_20261023';speed=point/'position_cache_e2e_parity_v1.json'
    if speed.exists():
        audit=json.loads(speed.read_text())
        for mode in ['before','after']:
            m=audit[mode];model=m['stages']['VGGT_model_only_s']
            for stage,values in {'rgb_to_delivered_ID':m['e2e'],**m['stages']}.items():
                fps_rows.append({'scope':'32_TRAIN_TIMESTAMPS_CONTROLLED_SPEED_ONLY','method':'A3_position_cache_'+mode,'stage':stage,
                    'timestamps':audit['frames'],'real_e2e_FPS':m['real_e2e_fps'] if stage=='rgb_to_delivered_ID' else 'NOT_APPLICABLE',
                    'VGGT_model_only_FPS':audit['frames']/model['total_s'] if stage=='VGGT_model_only_s' else 'NOT_APPLICABLE',
                    'P50_s':values['p50_s'],'P95_s':values['p95_s'],
                    'conditions':'exclusive single V100 GPU3; real live detector; FP16 Aggregator / FP32 TrackHead',
                    'source':str(speed)})
    with (output/'gmt_vggt_point_fps.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['scope','method','stage','timestamps','real_e2e_FPS','VGGT_model_only_FPS','P50_s','P95_s','conditions','source'])
        writer.writeheader();writer.writerows(fps_rows)
    diagnostic_files=['train_temporal_default_fullscene8_diagnostic_v2.json','train_joint_default_fullscene8_diagnostic_v2.json',
        'train_joint_registered_k16_c0.2_s0.1_first32_diagnostic_v1.json']
    diagnostics=[]
    for name in diagnostic_files:
        path=point/name
        if path.exists():
            d=json.loads(path.read_text());diagnostics.append({'artifact':str(path),'sha256':sha256_file(path),
                'scope':d['scope'],'counts':d['counts'],'official_TEST_metrics':False})
    attribution={'status':'TRAIN_DIAGNOSTICS_ONLY_FORMAL_TEST_ERROR_ATTRIBUTION_PENDING',
        'raw_GT_modified':False,'diagnostics':diagnostics,
        'engineering_failures_preserved':[
            {'failure':'bbox rectangle incorrectly represented as real instance mask','preserved_attempt':'gmt_detector_comparison_20261023/train_v1_visibility_audit2_v1',
             'fix_evidence':'results/gmt_bbox_visibility_evidence_audit_v2.json'},
            {'failure':'float64 xyxy subtraction versus FP32 xywh cache; 0.0000076294 pixel width delta',
             'preserved_attempt':'gmt_detector_comparison_20261023/train_v1_validation_shared_gpu0_v1/scene_04',
             'fix_evidence':'results/gmt_live_difference_scene4_598_v2.json'}],
        'scientific_limitations':['Default full TRAIN Scene8 A3 has 0/936 correct shared cross-view IDs.',
            'Early lower-gate results cover only an already registered 32-frame two-person TRAIN window.',
            'GT identity consistency of box support is not dense correspondence ground truth.',
            'Unmatched or ambiguous GT boxes cannot all be called detector false positives.',
            'Independent geometry identity benefit over shared-detector 2D controls remains unmeasured.']}
    (output/'gmt_vggt_point_error_attribution.json').write_text(json.dumps(attribution,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'official_methods_available':[r['Method'] for r in rows if r['HOTA']!='NOT_AVAILABLE'],
        'new_methods_not_yet_measured':[r['Method'] for r in rows if r['HOTA']=='NOT_AVAILABLE']}))


if __name__=='__main__':main()
