"""Gate head-free prior evaluation on exact old-control and zero outputs."""
from common import *


def main():
    specs=[('native_prior_prefix40_scene1_v1','constant_gmt_evidence_scene1_full_v1',40),
           ('native_prior_zero_prefix60_scene1_v1','online_B0_full_v1_source44_v1',60)]
    data=train_data();rows=[]
    for candidate_tag,reference_tag,n in specs:
        candidate,reference=[RUNTIME/'online'/tag for tag in (candidate_tag,reference_tag)]
        cm,rm=[json.loads((path/'manifest.json').read_text()) for path in (candidate,reference)]
        for field in ('checkpoint_sha256','config_sha256','annotation_sha256','seed','video_id','causal_online'):
            if cm[field]!=rm[field]:raise ValueError('unpaired prefix: '+field)
        if cm['full_video'] or cm['synchronized_groups']!=n or not rm['full_video']:raise ValueError('wrong prefix/full scope')
        if cm['total_vggt_calls'] or cm['dinov2_calls'] or cm['native_prior_extra_encoder_or_head_loaded'] or cm['native_prior_extra_learned_parameters']:
            raise ValueError('unexpected extra encoder/head in fixed prior')
        stats=cm['native_prior_observation_stats']
        if stats['maximum_extra_vector_columns'] or stats['observed_groups']!=(n if cm['feature_weight'] else 0):
            raise ValueError('unexpected feature payload or observed groups')
        for path,manifest in ((candidate,cm),(reference,rm)):
            if sha256(path/'predictions.json')!=manifest['predictions_sha256']:raise ValueError('changed predictions')
        ids={row['id'] for row in data['images'] if row['video_id']==1 and row['frame_id']<=n}
        actual=json.loads((candidate/'predictions.json').read_text())
        expected=[row for row in json.loads((reference/'predictions.json').read_text()) if row['image_id'] in ids]
        if actual!=expected:raise ValueError('fixed prior or zero control changes expected identity prefix')
        rows.append(dict(candidate=str(candidate),reference=str(reference),groups=n,exact_serialized_predictions=True,
                         manifest_sha256=[sha256(path/'manifest.json') for path in (candidate,reference)],
                         actual_extra_encoder_calls=0,additional_learned_parameters=0))
    save_json(RUNTIME/'matrix_reports/native_prior_prefix_gate_v1.json',dict(status='PASS',rows=rows,
        scope='40-group head-free prior equals measured trained-head constant-score control; zero60-group equals original causal GMT; no full-video or full-DEV conclusion inferred',provenance=provenance()))


if __name__=='__main__':main()
