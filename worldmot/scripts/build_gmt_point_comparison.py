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
    print(json.dumps({'status':report['status'],'official_methods_available':[r['Method'] for r in rows if r['HOTA']!='NOT_AVAILABLE'],
        'new_methods_not_yet_measured':[r['Method'] for r in rows if r['HOTA']=='NOT_AVAILABLE']}))


if __name__=='__main__':main()
