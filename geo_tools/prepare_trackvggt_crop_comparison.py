"""One bounded, GT/teacher-independent subset for full/local/combined tests."""
import collections
from common import *


def select_fit(items,n=256):
    scenes=collections.defaultdict(list)
    for row in items:scenes[row['scene']].append(row)
    for scene in scenes:scenes[scene].sort(key=lambda row:(hashlib.sha256(json.dumps(row['key']).encode()).hexdigest(),row['key']))
    selected=[]
    while len(selected)<min(n,len(items)):
        for scene in sorted(scenes):
            if scenes[scene] and len(selected)<n:selected.append(scenes[scene].pop(0))
    return selected


def main():
    runtime=Path('/data1/liuyeqiang/WWW_trackvggt_runtime');base=runtime/'cohorts/hard_events_expanded_v3/cohort.json'
    cohort=json.loads(base.read_text());target=ROOT/'manifests/trackvggt_crop_comparison_v1.json'
    if target.exists():raise FileExistsError(target)
    fit=select_fit(cohort['fit']);validation=cohort['validation']
    plan=dict(status='LOCKED_OBSERVATION_ONLY_CROP_COMPARISON',base_cohort=str(base),base_cohort_sha256=sha256(base),
        fit_keys=[row['key'] for row in fit],validation_keys=[row['key'] for row in validation],fit_queries=len(fit),validation_queries=len(validation),
        fit_scene_counts=dict(collections.Counter(str(r['scene']) for r in fit)),
        selection='Stable physical-key SHA256 order within each scene, scene round robin256 fit; unchanged original300 validation. No GT, patch-bin, error, teacher-score or new DEV12-release selection.',
        resolution=224,context_expansion=1.5,maximum_crop_tokens=32,crop_inputs='One current native-box crop from its actual physical image, aspect-preserving white pad224. A local magnification, not a new physical camera. No pose/depth outputs.',
        prepool='Union of inside-spatial24 and attention/spatial8; <=32 tokens, <=8 context. Same rule independent VGGT/DINO. Original-coordinate centers retained.',
        comparison=['full','local','combined'],model='Same AG-C k24, mean history, seed20261015, batch4, AdamW1e-4/wd.01, pilot150->1000 total; all three variants same256 fit/300 validation. Original full3000 fit comparison is separate.',
        backbone_controls=['vggt','dinov2'],feature_manifests={b:str(runtime/'tokens'/f'{b}_crop224_v1/manifest.json') for b in ('vggt','dinov2')},
        shared_NPZ_cap_bytes=2*(1<<30),maximum_new_frozen_encoder_GPU_wall_hours=1.,
        no_dense_maps_or_images_saved=True,new_DEV12_release_not_consumed=True,
        limits='One prespecified224 crop scale. No336 escalation from validation results, no new detector or candidates, no GT-generated crop/track, no final teacher/student claim from this repair.')
    save_json(target,plan);print(json.dumps(dict(plan=str(target),sha256=sha256(target),fit=len(fit),validation=len(validation))))


if __name__=='__main__':main()
