"""Actual seeded-wrapper prefix must preserve the primary reference first."""
import argparse
from common import *
from audit_dense_geometry_ablation import shared_options
from audit_suite_settings import signature


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--report-output',type=Path,required=True);a=p.parse_args()
    run=json.loads((a.run/'manifest.json').read_text());reference=json.loads((a.reference.parent/'manifest.json').read_text())
    if run['full_video'] or not reference['full_video'] or not run['causal_online'] or not reference['causal_online']:raise ValueError('actual causal prefix and complete reference required')
    for field in ('seed','explicit_inference_seed'):
        if run[field]!=reference['seed']:raise ValueError('same-seed wrapper gate required')
    actual=json.loads((a.run/'predictions.json').read_text());n=run['synchronized_groups']
    ids={row['image_id'] for row in video_records(train_data(),reference['video_id'],n)}
    if run['images']!=len(ids) or run['total_vggt_calls']!=len(ids) or run['geo_failures']:raise ValueError('input/actual encoder scope mismatch')
    if signature(run)!=signature(reference) or shared_options(run)!=shared_options(reference):raise ValueError('confounded seeded-wrapper configuration')
    if run['geo_options']['GEOMETRY_WEIGHT']!=reference['geo_options']['GEOMETRY_WEIGHT']:raise ValueError('geometry coefficient changed')
    if sha256(a.run/'predictions.json')!=run['predictions_sha256'] or sha256(a.reference)!=reference['predictions_sha256']:raise ValueError('changed predictions')
    expected=[row for row in json.loads(a.reference.read_text()) if row['image_id'] in ids]
    equal=actual==expected
    save_json(a.report_output,dict(status='PASS' if equal else 'FAIL',full_video=False,exact_serialized_predictions=equal,
        run=str(a.run),reference=str(a.reference),groups=n,actual_encoder_calls=run['total_vggt_calls'],
        manifest_sha256=[sha256(a.run/'manifest.json'),sha256(a.reference.parent/'manifest.json')],
        scope='actual primary-head same-inference-seed prefix wrapper gate; separate RNG-seed full evaluations do not inherit outcome equality',provenance=provenance()))
    if not equal:raise SystemExit('seeded wrapper changes primary same-seed prefix')


if __name__=='__main__':main()
