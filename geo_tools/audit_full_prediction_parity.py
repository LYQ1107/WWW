"""Exact complete strict5 OFF/zero output parity against the locked baseline."""
import argparse
from common import *


def load_prediction(path):
    info=path.with_name(path.stem+'_manifest.json')
    if not info.exists():info=path.parent/'manifest.json'
    report=json.loads(info.read_text())
    if sha256(path)!=report.get('prediction_sha256',report.get('predictions_sha256')):
        raise ValueError('prediction hash changed')
    if not report['full_video']:raise ValueError('complete videos required')
    return json.loads(path.read_text()),report,info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-sources',type=Path,required=True)
    parser.add_argument('--candidate-sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    specs=[json.loads(p.read_text())['video_predictions'] for p in (args.baseline_sources,args.candidate_sources)]
    required=set(map(str,json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids']))
    if any(set(spec)!=required for spec in specs):raise ValueError('every declared strict scene required')
    rows=[]
    for video in sorted(required,key=int):
        paths=[Path(spec[video]) for spec in specs]
        (base,bm,bp),(candidate,cm,cp)=[load_prediction(path) for path in paths]
        for field in ('causal_online','checkpoint_sha256','config_sha256','annotation_sha256','seed','video_id'):
            if bm[field]!=cm[field]:raise ValueError('unpaired inference: '+field)
        for report in (bm,cm):
            if report['video_id']!=int(video):raise ValueError('source scene index mismatch')
            if report.get('validation_corrected_gmt_camera_index',False):raise ValueError('camera repair is a separate control')
            opts=report.get('geo_options') or {}
            if opts.get('ENABLED',False) and (opts.get('FEATURE_WEIGHT',0) or opts.get('GEOMETRY_WEIGHT',0)):
                raise ValueError('OFF/zero control required')
        rows.append(dict(video_id=int(video),exact_serialized_predictions=base==candidate,
                         baseline_detections=len(base),candidate_detections=len(candidate),
                         prediction_paths=list(map(str,paths)),prediction_sha256=list(map(sha256,paths)),
                         run_manifest_sha256=[sha256(bp),sha256(cp)],source_commits=[bm['commit'],cm['commit']]))
    ok=all(row['exact_serialized_predictions'] for row in rows)
    save_json(args.output,dict(status='PASS' if ok else 'FAIL',full_strict_suite=True,rows=rows,
        scope='exact serialized detector boxes, scores, categories, image IDs, track IDs and output row order on every complete strict development scene; internal candidate order has separate implementation tests; no renaming or mean-metric substitute',
        source_specs_sha256=[sha256(p) for p in (args.baseline_sources,args.candidate_sources)],provenance=provenance()))
    if not ok:raise SystemExit('full output parity differs')


if __name__=='__main__':main()
