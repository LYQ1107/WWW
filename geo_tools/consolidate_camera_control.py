"""Consolidate independently checked complete camera-index controls."""
from common import *


def main():
    rows=[]
    strict=sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    for video in strict:
        path=ROOT/'results'/f'camera_index_corrected_full_v{video}_audit_v2.json'
        report=json.loads(path.read_text())
        if report['status']!='PASS' or report['video_id']!=video:raise ValueError('invalid scene audit')
        for source,digest in report['artifact_sha256'].items():
            if sha256(source)!=digest:raise ValueError('changed source artifact')
        fields=('exact_serialized_predictions','identical_detector_membership','exact_detector_row_order',
                'changed_track_id_rows','same_identity_partition_up_to_global_bijection','pure_gmt_evidence','metric_delta')
        rows.append(dict(video_id=video,source_audit=str(path),source_sha256=sha256(path),**{key:report[key] for key in fields}))
    if not all(r['identical_detector_membership'] for r in rows):raise ValueError('detector membership changed')
    pair_path=ROOT/'results/camera_index_corrected_full_strict_identity_pair_audit_v2.json'
    pair=json.loads(pair_path.read_text())
    if pair['status']!='PASS' or not pair['full_strict_suite']:raise ValueError('incomplete identity relation audit')
    metric_paths=[ROOT/'results'/f'{name}_metrics.json' for name in ('causal_B0_strict_suite_v1','camera_index_corrected_B0_strict_suite_v1')]
    metrics=[json.loads(path.read_text()) for path in metric_paths]
    if any(m['status']!='PASS' or m['video_ids']!=strict for m in metrics):raise ValueError('incomplete native suite')
    save_json(ROOT/'results/camera_index_corrected_full_strict_audit_v2.json',dict(
        status='EXACT_PREDICTION_PARITY' if all(r['exact_serialized_predictions'] for r in rows) else 'PREDICTION_DIFFERENCES_OBSERVED',
        full_strict_suite=True,rows=rows,identical_aggregate_metrics=metrics[0]['metrics']==metrics[1]['metrics'],
        metric_artifacts={str(path):sha256(path) for path in metric_paths},pair_audit=str(pair_path),pair_audit_sha256=sha256(pair_path),
        scope='all five complete causal strict scenes; every serialized image/box/score/class/track ID and row order; paired frozen checkpoint, YAML, annotation, seed and counters',
        limitations='same aggregate metrics do not imply identical predictions or partitions; live spatial-MLP hook is checked separately; legacy OFF18 absent options are verified by exact captured driver hash, not assumed zero',provenance=provenance()))


if __name__=='__main__':main()
