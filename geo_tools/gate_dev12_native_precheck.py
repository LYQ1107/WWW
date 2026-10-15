"""Gate actual export-off/on runs and record finite independent resource cost."""
import argparse
from common import *
from validate_dev12_native_inputs import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    directory=a.directory;off=directory/'off40_v1';on=directory/'on40_v1'
    reports=[json.loads((path/'manifest.json').read_text()) for path in (off,on)]
    if any(r['status']!='CAPTURE_COMPLETE_PARITY_PASS_UNLABELED' or r['groups']!=40 for r in reports):raise ValueError('real40 incomplete')
    if reports[0]['scope_sha256']!=reports[1]['scope_sha256']:raise ValueError('scope mismatch')
    if (off/'predictions.json').read_bytes()!=(on/'predictions.json').read_bytes():raise ValueError('export altered predictions')
    if (off/'group_state_fingerprints.json').read_bytes()!=(on/'group_state_fingerprints.json').read_bytes():raise ValueError('export altered original Gallery/Bank state')
    consumer=validate(on);save_json(directory/'on40_consumer.json',consumer)
    # Deliberately conservative extrapolation, and account complete outer
    # subprocess wall times (including import/loading/hash/export), not FPS.
    ledger=json.loads((directory/'resource_ledger.json').read_text())
    consumed=sum(r['outer_seconds'] for r in ledger['runs'])/3600
    measured=next(r['outer_seconds'] for r in ledger['runs'] if r['name']=='on40_v1')
    estimate=measured*30*1.8/3600
    scope=json.loads(Path(reports[0]['scope']).read_text())
    if consumed+estimate>scope['maximum_total_gpu_wall_hours']:raise ValueError('full run exceeds independent budget')
    result=dict(status='PASS',scope_sha256=reports[0]['scope_sha256'],export_off_vs_original_predictions_exact=True,
        export_on_vs_original_predictions_exact=True,off_vs_on_prediction_bytes_exact=True,all40_original_Gallery_Bank_state_fingerprints_exact=True,
        consumer=consumer,prechecks_outer_gpu_wall_hours=consumed,conservative_full_outer_gpu_wall_hours=estimate,
        maximum_total_gpu_wall_hours=scope['maximum_total_gpu_wall_hours'],original_sealed_predictions_reference=reports[0]['predictions_equal_sealed_native'],
        inputs_and_full_histories_verified=True,GT_read=False,known_errors_not_used_for_scope=True,
        source_manifests={str(path/'manifest.json'):sha256(path/'manifest.json') for path in (off,on)},provenance=provenance())
    save_json(directory/'gate.json',result);print(json.dumps({k:v for k,v in result.items() if k!='provenance'}))


if __name__=='__main__':main()
