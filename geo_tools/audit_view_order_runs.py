"""Verify full camera-order tracking behavior; never infer valid pose parity."""
import argparse
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--normal-sources',type=Path,required=True);parser.add_argument('--reversed-sources',type=Path,required=True);parser.add_argument('--tag',required=True);args=parser.parse_args()
    sources=[json.loads(path.read_text())['video_predictions'] for path in [args.normal_sources,args.reversed_sources]]
    vids=sorted(json.loads((ROOT/'manifests/geo_split_v1.json').read_text())['strict_development_video_ids'])
    if any(sorted(map(int,source))!=vids for source in sources):raise ValueError('complete strict5 required')
    rows=[]
    for video in vids:
        paths=[Path(source[str(video)]) for source in sources]
        runs=[json.loads((path.parent/'manifest.json').read_text()) for path in paths]
        if not all(run['full_video'] and run['causal_online'] and run['video_id']==video for run in runs):raise ValueError('complete paired causal video required')
        if runs[0].get('validation_reversed_vggt_order',False) or not runs[1].get('validation_reversed_vggt_order',False):raise ValueError('normal/reversed roles invalid')
        for key in ['checkpoint_sha256','config_sha256','head_checkpoint_sha256','geo_options','images','synchronized_groups']:
            if runs[0][key]!=runs[1][key]:raise ValueError(f'configuration/input mismatch: {key}')
        if any(sha256(path)!=run['predictions_sha256'] for path,run in zip(paths,runs)):raise ValueError('changed prediction artifacts')
        rows.append(dict(video_id=video,normal_prediction=str(paths[0]),reversed_prediction=str(paths[1]),prediction_sha256=[sha256(path) for path in paths],
            exact_predictions=json.loads(paths[0].read_text())==json.loads(paths[1].read_text()),actual_encoder_calls=[run['total_vggt_calls'] for run in runs],
            valid_geometry_calls=[run['geometry_valid_calls'] for run in runs],failures=[run['geo_failures'] for run in runs],
            normal_manifest_sha256=sha256(paths[0].parent/'manifest.json'),reversed_manifest_sha256=sha256(paths[1].parent/'manifest.json')))
    save_json(ROOT/'results'/f'{args.tag}.json',dict(status='EXACT_TRACKING_PARITY' if all(row['exact_predictions'] for row in rows) else 'TRACKING_DIFFERENCES_OBSERVED',
        scope='all strict complete causal videos; VGGT input order swapped and physical camera IDs restored; canonical GMT order identical',rows=rows,
        limitations='identical tracking under rejected geometry does not prove valid-pose prediction order invariance; no arbitrary GMT query-order invariance claim',provenance=provenance()))


if __name__=='__main__':main()
