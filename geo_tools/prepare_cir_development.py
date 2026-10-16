"""Freeze the already-declared full video split; no result-driven scope choice."""
import argparse
import json
from pathlib import Path
from common import ROOT,sha256,save_json
from cir_configuration import saved_configuration_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True)
    p.add_argument('--c1',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    original=json.loads(a.original.read_text());c1=json.loads(a.c1.read_text())
    if original['status']!='COMPLETE' or c1['status']!='COMPLETE' or not original['full_video'] or not c1['full_video']:
        raise ValueError('Both complete actual-image engineering runs required')
    if original['video_id']!=4 or c1['video_id']!=4 or c1['method']!='c1':raise ValueError('Frozen engineering scope mismatch')
    if c1['CIR_stats']['fallback_reasons'] or c1['CIR_stats']['extra_joint_solves']<1 or c1['CIR_stats']['changed_primary_rows']<1:
        raise ValueError('Actual C1 mutated-state path not qualified')
    if c1['VGGT_calls'] or c1['DINO_calls'] or c1['new_learnable_parameters'] or c1['source_tree_dirty_at_video_start']:
        raise ValueError('Unqualified extra model or mutable source')
    for kind in ('raw','stock'):
        for run in (original,c1):
            spec=run['predictions'][kind]
            if sha256(spec['path'])!=spec['sha256']:raise ValueError('Actual trajectory changed')
    def signature(run):
        rows=json.loads(Path(run['predictions']['raw']['path']).read_text())
        return [(r['image_id'],r['bbox'],r['score'],r['category_id']) for r in rows]
    if signature(original)!=signature(c1) or original['image_sha256']!=c1['image_sha256']:raise ValueError('Perception/input changes')
    left=saved_configuration_hashes(a.original.parent/'config.yaml',original['checkpoint_sha256'])
    right=saved_configuration_hashes(a.c1.parent/'config.yaml',c1['checkpoint_sha256'])
    if left['frozen_GMT_config_sha256']!=right['frozen_GMT_config_sha256']:raise ValueError('Non-CIR config changed')
    split_path=ROOT/'manifests/geo_split_v1.json';split=json.loads(split_path.read_text())
    required=[1,2,3,10,11,14,15,16,18,19,22,23]
    if split['full_development_video_ids']!=required or split['strict_development_video_ids']!=[1,10,11,18,19]:raise ValueError('Original split drift')
    core={name:sha256(ROOT/name) for name in ('gtr/config.py','gtr/modeling/meta_arch/gtr_rcnn.py',
        'gtr/geometry/online_runner.py','gtr/modeling/identity_revision/controller.py')}
    if any(c1['source_sha256'].get(name)!=digest for name,digest in core.items()):raise ValueError('Qualified algorithm changed')
    gate=dict(status='PASS_FULL_VIDEO_NATIVE_OFF_QUALIFICATION',stage='P1_REGISTERED_COMPLETE_DEVELOPMENT',
        current_source_sha256=core,allowed_methods=['original','off','c1'],allowed_video_ids=required,
        full_registered_development_video_ids=required,strict_metrics_video_ids=[1,10,11,18,19],
        raw_GT_invalid_video_ids_not_dropped=[2,3,14,15,16,22,23],
        split_sha256=sha256(split_path),protocol_sha256=sha256(ROOT/'docs/CIR_GMT/TRAIN24_PROTOCOL.md'),
        engineering_runs={str(p):sha256(p) for p in (a.original,a.c1)},
        frozen_GMT_config_sha256=left['frozen_GMT_config_sha256'],C1_alpha=2.,
        first_group_and_Bank='unchanged',modes=['Original GMT','GMT+C1'],
        scientific_success=False,development_scores_read_for_this_gate=False,
        selection='same complete12 declared before any CIR result; duplicate GT events retained, unreliable labels masked and strict metric blocks disclosed',
        next_phase='minimum CIR engineering continues regardless of C1 quality sign')
    save_json(a.out,gate);print(json.dumps(gate),flush=True)


if __name__=='__main__':main()
