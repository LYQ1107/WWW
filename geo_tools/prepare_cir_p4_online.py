"""Source/checkpoint-bound P4 native admission, without reading GT or metrics."""
import argparse
import json
from pathlib import Path
from common import ROOT,sha256,save_json
from cir_configuration import saved_configuration_hashes


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--off-prefix',type=Path,required=True)
    p.add_argument('--policy-prefix',type=Path,action='append',required=True)
    p.add_argument('--fault-proof',type=Path,action='append',required=True)
    p.add_argument('--training-report',type=Path,required=True);p.add_argument('--original-full',type=Path,required=True)
    p.add_argument('--off-full',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    protocol=read(ROOT/'results/CIR_GMT/P4_NATIVE_INTEGRATION_PROTOCOL.json');fit=read(a.training_report)
    if fit['status']!='COMPLETE_FIRST_SEED_FIT12_RISK_ONLY_NATIVE_ONLINE_PENDING':raise ValueError('Actual frozen FIT training required')
    off=read(a.off_prefix)
    if off['status']!='PASS_REAL_PREFIX_OFF_KEEP_ROLLBACK' or not all(all(v.values()) for v in off['comparisons'].values()):
        raise ValueError('Current OFF/KEEP and native rollback qualification required')
    core={k:v for k,v in off['source_sha256_at_import'].items() if k.startswith('gtr/')}
    def pin(proof):
        for name,digest in core.items():
            if sha256(ROOT/name)!=digest or proof['source_sha256_at_import'].get(name)!=digest:raise ValueError('Native source drift: '+name)
    pin(off);evidence=[a.off_prefix,a.training_report,a.original_full];modes=set();models={}
    for path in a.policy_prefix:
        proof=read(path);mode=proof['policy'];pin(proof)
        if mode not in ('guard_reference','risk_mlp','risk_grouped') or proof['status']!=f'PASS_REAL_PREFIX_{mode.upper()}_OFFICIAL_CAUSAL':raise ValueError('Native policy prefix failed')
        if not all(all(v.values()) for v in proof['comparisons'].values()):raise ValueError('Official/causal native mismatch')
        modes.add(mode);evidence.append(path)
        if mode.startswith('risk_'):
            frozen=protocol['learned_models_frozen'][mode];model=fit['models'][frozen['architecture']];cp=model['checkpoint']
            if cp['sha256']!=frozen['checkpoint_sha256'] or sha256(cp['path'])!=cp['sha256'] or proof['risk_checkpoint_sha256']!=cp['sha256']:
                raise ValueError('Frozen checkpoint provenance mismatch')
            if proof['risk_trace_sha256']!=sha256(ROOT/'geo_tools/cir_risk_trace.py'):raise ValueError('Risk observer source drift')
            if not all(v['model_forward_calls']>0 and v['evidence_calls']>0 for v in proof['risk_checks'].values()):raise ValueError('Actual neural evidence/forward not exercised')
            models[mode]=dict(path=cp['path'],sha256=cp['sha256'],threshold=frozen['threshold'],architecture=frozen['architecture'])
    if modes!={'guard_reference','risk_mlp','risk_grouped'}:raise ValueError('All fixed P4 policies required')
    faults=set()
    for path in a.fault_proof:
        proof=read(path);pin(proof);mode=proof['policy']
        if mode not in models or proof['status']!='PASS_REAL_GUARD_COMMIT_ROLLBACK_FUTURE' or not all(proof['checks'].values()):raise ValueError('Real frozen-head native fault recovery failed')
        if proof['risk_checkpoint_sha256']!=models[mode]['sha256'] or not proof['checks'].get('frozen_head_restored') or not proof['checks'].get('actual_frozen_head_forward_exercised'):
            raise ValueError('Actual checkpoint/fault forward not proved')
        faults.add(mode);evidence.append(path)
    if faults!=set(models):raise ValueError('Both actual checkpoint fault recoveries required')
    original=read(a.original_full)
    if original['status']!='COMPLETE' or not original['full_video'] or original['video_id']!=4 or original['method']!='original':raise ValueError('Complete original engineering reference required')
    allowed=['off'];videos=[4]
    if a.off_full:
        current=read(a.off_full)
        if current['status']!='COMPLETE' or not current['full_video'] or current['method']!='off' or current['video_id']!=4 or current['source_tree_dirty_at_video_start']:
            raise ValueError('Clean current complete OFF required')
        for name,digest in core.items():
            if name!='gtr/geometry/decision_trace.py' and current['source_sha256'].get(name)!=digest:raise ValueError('Complete OFF source mismatch')
        for kind in ('raw','stock'):
            left=current['predictions'][kind];right=original['predictions'][kind]
            if sha256(left['path'])!=left['sha256'] or sha256(right['path'])!=right['sha256'] or read(left['path'])!=read(right['path']):raise ValueError('Current complete OFF outputs differ')
        hashes=[saved_configuration_hashes(path.parent/'config.yaml',run['checkpoint_sha256'])['frozen_GMT_config_sha256'] for path,run in ((a.original_full,original),(a.off_full,current))]
        if current['image_sha256']!=original['image_sha256'] or len(set(hashes))!=1:raise ValueError('Actual images/config changed')
        allowed+=['guard_reference','risk_mlp','risk_grouped'];videos=[1,2,3,10,11,14,15,16,18,19,22,23];evidence.append(a.off_full)
    save_json(a.out,dict(status='PASS_FULL_VIDEO_NATIVE_OFF_QUALIFICATION',stage='P4_FIRST_SEED_RISK_NATIVE',
        current_source_sha256=core,allowed_methods=allowed,allowed_video_ids=videos,risk_models=models,
        native_evidence=[dict(path=str(path),sha256=sha256(path)) for path in evidence],
        scientific_success=False,GT_read=False,metrics_read=False,full24_fit_complete=False,
        scope='First-seed complete DEV qualification only; no TEST or independent architecture-gain claim'))


if __name__=='__main__':main()
