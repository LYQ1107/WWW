"""Qualify the finite preregistered full-video guard/attribution controls.

No metric values or GT are consulted; all source and actual OFF outputs bind.
"""
import argparse
import json
from pathlib import Path
from common import ROOT,sha256,save_json
from cir_configuration import saved_configuration_hashes


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--off-prefix',type=Path,required=True)
    p.add_argument('--guard-prefix',type=Path,required=True);p.add_argument('--original-full',type=Path,required=True)
    p.add_argument('--off-full',type=Path);p.add_argument('--control-prefix',type=Path,action='append',default=[])
    p.add_argument('--rollback',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    off=read(a.off_prefix);guard=read(a.guard_prefix)
    if off['status']!='PASS_REAL_PREFIX_OFF_KEEP_ROLLBACK' or guard['status']!='PASS_REAL_PREFIX_GUARD_OFFICIAL_CAUSAL':
        raise ValueError('Current original/OFF/KEEP and Guard shared-core proofs required')
    core={name:digest for name,digest in off['source_sha256_at_import'].items() if name.startswith('gtr/')}
    def pin(proof):
        for name,digest in core.items():
            if sha256(ROOT/name)!=digest or proof['source_sha256_at_import'].get(name)!=digest:
                raise ValueError('Qualified native source drift: '+name)
    pin(off);pin(guard)
    for proof in (off,guard):
        if not all(all(c.values()) for c in proof['comparisons'].values()):raise ValueError('Native parity failed')
    original=read(a.original_full)
    if original['status']!='COMPLETE' or not original['full_video'] or original['video_id']!=4 or original['method']!='original':
        raise ValueError('Actual complete engineering original required')
    allowed=['off'];videos=[4];evidence=[a.off_prefix,a.guard_prefix,a.original_full]
    if a.off_full:
        current=read(a.off_full)
        if current['status']!='COMPLETE' or not current['full_video'] or current['method']!='off' or current['video_id']!=4:
            raise ValueError('Current complete OFF actor required')
        if current['source_tree_dirty_at_video_start']:raise ValueError('Full OFF source must be clean')
        for name,digest in core.items():
            if current['source_sha256'].get(name)!=digest:raise ValueError('Full OFF source mismatch: '+name)
        for kind in ('raw','stock'):
            left=current['predictions'][kind];right=original['predictions'][kind]
            if sha256(left['path'])!=left['sha256'] or sha256(right['path'])!=right['sha256']:
                raise ValueError('Actual output drift')
            if read(left['path'])!=read(right['path']):raise ValueError('Complete OFF output differs from original')
        if current['image_sha256']!=original['image_sha256']:raise ValueError('Real image inputs differ')
        configs=[saved_configuration_hashes(path.parent/'config.yaml',run['checkpoint_sha256'])['frozen_GMT_config_sha256']
                 for path,run in ((a.off_full,current),(a.original_full,original))]
        if len(set(configs))!=1:raise ValueError('Frozen GMT configuration differs')
        modes=set()
        for path in a.control_prefix:
            proof=read(path);mode=proof['policy'];pin(proof)
            if proof['status']!=f'PASS_REAL_PREFIX_{mode.upper()}_OFFICIAL_CAUSAL':raise ValueError('Control prefix failed')
            if not all(all(v.values()) for v in proof['comparisons'].values()):raise ValueError('Control prefix mismatch')
            modes.add(mode);evidence.append(path)
        if modes!={'margin','threshold','bias2'}:raise ValueError('All registered controls require native prefix proof')
        if not a.rollback:raise ValueError('Real guarded commit rollback proof required')
        rollback=read(a.rollback)
        if rollback['status']!='PASS_REAL_GUARD_COMMIT_ROLLBACK_FUTURE' or not all(rollback['checks'].values()):
            raise ValueError('Real guarded commit rollback failed')
        pin(rollback)
        allowed=['off','guard','margin','threshold','bias2'];videos=[1,2,3,10,11,14,15,16,18,19,22,23]
        evidence.extend([a.off_full,a.rollback])
    save_json(a.out,dict(status='PASS_FULL_VIDEO_NATIVE_OFF_QUALIFICATION',stage='P2_FINITE_CONTROLS',
        allowed_methods=allowed,allowed_video_ids=videos,current_source_sha256=core,
        native_evidence=[dict(path=str(path),sha256=sha256(path)) for path in evidence],
        scientific_success=False,metrics_used_for_admission=False,GT_in_inference=False,
        interpretation='Engineering gate only. Full12 inference required; strict metrics retain all raw-GT validity blocks. No TEST permission from this gate.'))


if __name__=='__main__':main()
