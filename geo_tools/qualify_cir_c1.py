"""Source-pinned staged eligibility: prefix -> complete OFF -> C1 engineering."""
import argparse
import json
from pathlib import Path
from common import ROOT,sha256,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--off-prefix',type=Path,required=True)
    p.add_argument('--c1-prefix',type=Path,required=True);p.add_argument('--original-full',type=Path,required=True)
    p.add_argument('--off-full',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    p0=json.loads((ROOT/'results/CIR_GMT/P0_NATIVE_QUALIFICATION.json').read_text())
    if p0['status']!='PASS_FULL_VIDEO_NATIVE_OFF_QUALIFICATION':raise ValueError('Original P0 must remain qualified')
    off=json.loads(a.off_prefix.read_text());c1=json.loads(a.c1_prefix.read_text())
    if off['status']!='PASS_REAL_PREFIX_OFF_KEEP_ROLLBACK' or c1['status']!='PASS_REAL_PREFIX_C1_OFFICIAL_CAUSAL':raise ValueError('Current core prefix failed')
    if not all(all(v.values()) for v in off['comparisons'].values()) or not all(all(v.values()) for v in c1['comparisons'].values()):raise ValueError('Current core prefix mismatch')
    sources={}
    for name in p0['current_source_sha256']:
        digest=sha256(ROOT/name)
        if off['source_sha256_at_import'][name]!=digest or c1['source_sha256_at_import'][name]!=digest:raise ValueError('Core source drift')
        sources[name]=digest
    original=json.loads(a.original_full.read_text())
    if original['status']!='COMPLETE' or not original['full_video'] or original['video_id']!=4:raise ValueError('Original complete official driver reference required')
    path=Path(original['predictions']['raw']['path'])
    if sha256(path)!=original['predictions']['raw']['sha256']:raise ValueError('Original output drift')
    prior=Path(p0['native_result']['path']).parent/'ORIGINAL_SOURCE/predictions.json'
    if json.loads(path.read_text())!=json.loads(prior.read_text()):raise ValueError('Complete official versus complete causal original mismatch')
    allowed=['off'];full_pass=False
    if a.off_full:
        result=json.loads(a.off_full.read_text())
        if result['status']!='COMPLETE' or not result['full_video'] or result['method']!='off' or result['video_id']!=4:raise ValueError('Current complete OFF missing')
        for kind in ('raw','stock'):
            spec=result['predictions'][kind];expected=original['predictions'][kind]
            if sha256(spec['path'])!=spec['sha256'] or sha256(expected['path'])!=expected['sha256']:raise ValueError('Full prediction drift')
            if json.loads(Path(spec['path']).read_text())!=json.loads(Path(expected['path']).read_text()):raise ValueError('Current full OFF differs from original '+kind)
        if result['image_sha256']!=original['image_sha256'] or result['config_sha256']!=original['config_sha256']:raise ValueError('Actual inputs or config differ')
        if any(result['source_sha256'][k]!=v for k,v in sources.items()):raise ValueError('Full OFF used different core source')
        allowed+=['c1'];full_pass=True
    refs={name:dict(path=str(value),sha256=sha256(value)) for name,value in (
        ('off_prefix',a.off_prefix),('c1_prefix',a.c1_prefix),('original_full',a.original_full))}
    if a.off_full:refs['off_full']=dict(path=str(a.off_full),sha256=sha256(a.off_full))
    gate=dict(status='PASS_FULL_VIDEO_NATIVE_OFF_QUALIFICATION',current_source_sha256=sources,
        allowed_methods=allowed,allowed_video_ids=[4],new_core_complete_OFF_passed=full_pass,
        parent_P0_sha256=sha256(ROOT/'results/CIR_GMT/P0_NATIVE_QUALIFICATION.json'),sources=refs,
        scientific_success=False,development_not_yet_admitted=True,
        interpretation='Full original/cold OFF source base preserved, current OFF/KEEP prefix and real C1 shared-core math qualified; C1 complete engineering run allowed only after current OFF full agreement')
    save_json(a.out,gate);print(json.dumps(gate),flush=True)


if __name__=='__main__':main()
